# Promptalpha

This application adapts the uploaded notebook's scoring and compression implementation into a Streamlit UI. It uses the same official LLMLingua-2 token-classification checkpoint, word alignment, chunking policy, `max` word/subtoken score combination, extractive compression logic, and tokenizer-based token counting.

## Reused notebook configuration

- Model: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- Model type: `AutoModelForTokenClassification`
- Word/token combination: `max`
- Maximum content tokens per model chunk: `510` (512 positions minus special tokens)
- Chunk boundaries: `.`, newline, and `;`, with notebook-style boundary backoff
- Keep score: `softmax(logits)[:, KEEP_LABEL_ID]`, with the notebook's keep-label resolution
- Token counts: the same scoring tokenizer with `add_special_tokens=False`
- Compression: every word with importance score `>= 0.25` is retained; original whitespace is formatted after extraction
- Score threshold used by the app: fixed at `0.25`
- Protected spans: text inside backticks, double quotes, or single quotes is retained as one unit without changing its model score; auxiliary-negation pairs such as `Do not` and `should not` keep both words together

The application uses an absolute score threshold, not a retention percentage. A word scored `0.25` or higher is retained, and no minimum output-token rule adds lower-scoring words. The scoring model itself remains unchanged.

## Installation

Python 3.10 or newer is recommended. Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

The first cleaning request downloads the model and tokenizer from Hugging Face. Ensure the machine has internet access and enough disk/RAM/VRAM. The model is cached by Transformers after the first download; Streamlit also caches the loaded model for the session.

## Run

From this directory:

```bash
streamlit run app.py
```

Paste a prompt, then use **Show Scores** to inspect the importance score, word text, and character span for every detected word. Use **Clean Prompt** to generate the compressed prompt and its token statistics.

## Validation performed

The implementation was checked for Python syntax and imports of the local modules. The full model inference was not run in this sandbox because it requires downloading the Hugging Face checkpoint and may require substantial resources. The app reports a user-facing processing error rather than fabricating output or statistics if model loading/inference fails.

## Limitations

The notebook's model was trained on meeting transcripts, so applying it to code-generation prompts has documented domain shift. The app performs word-level extractive compression, not generative rewriting. Long prompts are scored in independent chunks, as in the notebook, so cross-chunk context is limited. GPU use is automatic when CUDA is available; otherwise inference runs on CPU and may be slow.
