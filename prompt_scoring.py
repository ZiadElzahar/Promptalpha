"""Notebook-faithful LLMLingua-2 word scoring and prompt compression."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

SCORING_MODEL_ID = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
MAX_CONTENT_TOKENS_PER_CHUNK = 510
CHUNK_END_TOKENS = (".", "\n", ";")
WORD_COMBINE = "max"
# Keep language-agnostic code and quoted spans intact. Their scores remain
# unchanged; protection is applied only at selection time.
WORD_RE = re.compile(
    r"```.*?```|`[^`\n]*`|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|\w+(?:['’]\w+)?|[^\w\s]+",
    re.DOTALL,
)

# These words carry negation, exclusivity, or hard requirements. They are
# protected independently of the model score because deleting them can invert
# the requested behavior.
PROTECTED_TERMS = frozenset({
    "not", "no", "never", "none", "neither", "nor", "except", "unless",
    "only", "must", "required", "without", "avoid", "cannot", "can't",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "shouldn't", "wouldn't", "couldn't", "won't", "shan't",
})
NEGATION_AUXILIARIES = frozenset({
    "do", "does", "did", "should", "would", "could", "will", "can",
    "is", "are", "was", "were", "have", "has", "had",
})
NEGATION_WORDS = frozenset({"not", "never"})


def is_delimited_protected_span(word: str) -> bool:
    stripped = word.strip()
    return (
        (stripped.startswith("```") and stripped.endswith("```"))
        or (stripped.startswith("`") and stripped.endswith("`") and len(stripped) >= 2)
        or (stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2)
        or (stripped.startswith("'") and stripped.endswith("'") and len(stripped) >= 2)
    )


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: torch.device
    keep_label_id: int


def resolve_keep_label_id(model_config, explicit: int | None = None) -> int:
    """Use the notebook's keep-label resolution rule."""
    if explicit is not None:
        return int(explicit)
    for key, value in dict(getattr(model_config, "id2label", {}) or {}).items():
        if "keep" in str(value).lower() or "preserve" in str(value).lower():
            return int(key)
    return 1


def load_model(force_cpu: bool = False) -> ModelBundle:
    """Load the exact notebook model and fast tokenizer."""
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(SCORING_MODEL_ID)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("The scoring tokenizer is not a fast tokenizer; offset alignment cannot be used safely.")
    model = AutoModelForTokenClassification.from_pretrained(SCORING_MODEL_ID).to(device)
    model.eval()
    keep_label_id = resolve_keep_label_id(model.config)
    return ModelBundle(tokenizer, model, device, keep_label_id)


def split_words(text: str):
    """Return untouched word/operator spans using the notebook's regex."""
    if not isinstance(text, str):
        return []
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def assign_token_scores_to_words(words, token_offsets, token_scores, combine=WORD_COMBINE):
    assert combine in ("mean", "max", "first")
    buckets = [[] for _ in words]
    wi = 0
    for (ts, te), score in zip(token_offsets, token_scores):
        if te <= ts:
            continue
        while wi < len(words) and words[wi][2] <= ts:
            wi += 1
        wj = wi
        while wj < len(words) and words[wj][1] < te:
            if words[wj][2] > ts:
                buckets[wj].append(score)
            wj += 1
    result = []
    for bucket in buckets:
        if not bucket:
            result.append(None)
        elif combine == "mean":
            result.append(sum(bucket) / len(bucket))
        elif combine == "max":
            result.append(max(bucket))
        else:
            result.append(bucket[0])
    return result


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _is_boundary(text, words, index, chunk_end_tokens):
    _, start, end = words[index]
    return bool(words[index][0] and words[index][0][-1] in chunk_end_tokens) or (
        end < len(text) and text[end] == "\n"
    )


def chunk_word_ranges(text, words, tokenizer, max_content_tokens=MAX_CONTENT_TOKENS_PER_CHUNK):
    ranges = []
    start = 0
    while start < len(words):
        chunk_start = words[start][1]
        if _count_tokens(tokenizer, text[chunk_start:words[start][2]]) > max_content_tokens:
            ranges.append((start, start + 1, True))
            start += 1
            continue
        lo, hi = start + 1, len(words)
        while lo <= hi:
            mid = (lo + hi) // 2
            if _count_tokens(tokenizer, text[chunk_start:words[mid - 1][2]]) <= max_content_tokens:
                lo = mid + 1
            else:
                hi = mid - 1
        end = hi
        if end - start > 4:
            lookback = max(1, int((end - start) * 0.2))
            for index in range(end - 1, max(start, end - 1 - lookback) - 1, -1):
                if _is_boundary(text, words, index, set(CHUNK_END_TOKENS)):
                    end = index + 1
                    break
        ranges.append((start, end, False))
        start = end
    return ranges


@torch.inference_mode()
def _score_chunk(bundle: ModelBundle, chunk_text: str):
    max_len = MAX_CONTENT_TOKENS_PER_CHUNK + 2
    encoded = bundle.tokenizer(
        chunk_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    offsets = encoded["offset_mapping"][0].tolist()
    encoded.pop("offset_mapping")
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    logits = bundle.model(**encoded).logits
    keep_probs = torch.softmax(logits.float(), dim=-1)[0, :, bundle.keep_label_id]
    n_tokens = int(encoded["input_ids"].shape[1])
    covered_end = max((end for _, end in offsets if end > 0), default=0)
    return offsets, keep_probs.tolist(), n_tokens, covered_end


def score_prompt(text: str, bundle: ModelBundle) -> dict:
    """Score all words with the notebook's chunking and alignment implementation."""
    words = split_words(text)
    if not words:
        return {"words": [], "scores": [], "status": "skipped_empty", "truncated": False, "n_chunks": 0, "n_model_tokens": 0}
    ranges = chunk_word_ranges(text, words, bundle.tokenizer)
    all_scores = [None] * len(words)
    truncated = False
    n_model_tokens = 0
    for word_start, word_end, oversized in ranges:
        char_start, char_end = words[word_start][1], words[word_end - 1][2]
        offsets, probabilities, n_tokens, covered_end = _score_chunk(bundle, text[char_start:char_end])
        n_model_tokens += n_tokens
        local_words = [(word, start - char_start, end - char_start) for word, start, end in words[word_start:word_end]]
        local_scores = assign_token_scores_to_words(local_words, offsets, probabilities, WORD_COMBINE)
        for index, score in enumerate(local_scores):
            if score is not None:
                all_scores[word_start + index] = float(score)
        if oversized or covered_end < (char_end - char_start):
            truncated = True
    scores = [0.0 if score is None else score for score in all_scores]
    status = "success_truncated" if truncated else ("success_chunked" if len(ranges) > 1 else "success")
    return {"words": words, "scores": scores, "status": status, "truncated": truncated, "n_chunks": len(ranges), "n_model_tokens": n_model_tokens}


def select_kept_indices(entries, score_threshold: float):
    ranked = sorted(entries, key=lambda entry: (-float(entry["importance_score"]), entry["word_index"]))
    protected_indices = {
        entry["word_index"]
        for entry in entries
        if is_delimited_protected_span(entry["word"])
        or entry["word"].strip("` \"'").lower() in PROTECTED_TERMS
    }
    # Protect both sides of explicit negation phrases: "Do not", "should not",
    # "does not", etc. Protecting only "not" would change the remaining phrase.
    for index, entry in enumerate(entries[:-1]):
        current = entry["word"].strip("` ").lower()
        following = entries[index + 1]["word"].strip("` ").lower()
        if current in NEGATION_AUXILIARIES and following in NEGATION_WORDS:
            protected_indices.update((entry["word_index"], entries[index + 1]["word_index"]))
    selected = [
        entry for entry in ranked
        if float(entry["importance_score"]) >= score_threshold
        or entry["word_index"] in protected_indices
    ]
    if not selected and ranked:
        selected = ranked[:1]
    return {entry["word_index"] for entry in selected}


def _render_selected_words(text: str, entries) -> str:
    spans = sorted((entry["start_char"], entry["end_char"]) for entry in entries)
    output, previous = [], 0

    def normalized_gap(gap: str) -> str:
        gap = re.sub(r"[ \t\f\v]+", " ", gap)
        return re.sub(r"[ \t]*\n[ \t]*", "\n", gap)

    for start, end in spans:
        output.append(normalized_gap("".join(char for char in text[previous:start] if char.isspace())))
        output.append(text[start:end])
        previous = end
    output.append(normalized_gap("".join(char for char in text[previous:] if char.isspace())))
    return "".join(output).strip()


def compress_prompt(text: str, result: dict, score_threshold: float) -> str:
    """Keep every scored word at or above the absolute importance threshold."""
    entries = [
        {"word_index": i, "word": word, "start_char": start, "end_char": end, "importance_score": score}
        for i, ((word, start, end), score) in enumerate(zip(result["words"], result["scores"]))
    ]
    kept = select_kept_indices(entries, score_threshold)
    return _render_selected_words(text, [entry for entry in entries if entry["word_index"] in kept])


def token_count(bundle: ModelBundle, text: str) -> int:
    """Exact notebook method: scoring tokenizer, no special tokens."""
    return _count_tokens(bundle.tokenizer, text)
