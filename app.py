from __future__ import annotations

import streamlit as st

from prompt_scoring import (
    MIN_OUTPUT_TOKENS,
    compress_prompt,
    load_model,
    score_prompt,
    token_count,
)

SCORE_THRESHOLD = 0.25

st.set_page_config(page_title="Promptalpha", page_icon=None, layout="wide")

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()
st.title("Promptalpha")

prompt = st.text_area(
    "Original prompt",
    height=280,
    placeholder="Paste the prompt you want to clean...",
)

if st.button("Show Scores", type="secondary"):
    if not prompt or not prompt.strip():
        st.warning("Enter a non-empty prompt first.")
    else:
        try:
            with st.spinner(""):
                score_bundle = get_model()
                score_result = score_prompt(prompt, score_bundle)
        except Exception:
            st.error("The scores could not be calculated. Verify that the model can be loaded and try again.")
        else:
            st.session_state["score_result"] = score_result

if "score_result" in st.session_state:
    score_result = st.session_state["score_result"]
    st.subheader("Word Scores")
    score_rows = [
        {
            "Word": word,
            "Importance score": round(float(score), 4),
            "Start": start,
            "End": end,
        }
        for (word, start, end), score in zip(score_result["words"], score_result["scores"])
    ]
    st.dataframe(score_rows, use_container_width=True, hide_index=True)

if st.button("Clean Prompt", type="primary", use_container_width=True):
    if not prompt or not prompt.strip():
        st.warning("Enter a non-empty prompt before cleaning.")
    else:
        try:
            with st.spinner(""):
                bundle = get_model()
                result = score_prompt(prompt, bundle)
                cleaned = compress_prompt(
                    prompt,
                    result,
                    SCORE_THRESHOLD,
                    tokenizer=bundle.tokenizer,
                    minimum_tokens=MIN_OUTPUT_TOKENS,
                )
                original_tokens = token_count(bundle, prompt)
                cleaned_tokens = token_count(bundle, cleaned)
        except Exception:
            st.error(
                "The prompt could not be processed. Verify that the model can be downloaded and that "
                "the environment has enough memory, then try again."
            )
        else:
            removed = original_tokens - cleaned_tokens
            reduction = (removed / original_tokens * 100) if original_tokens else 0.0
            st.session_state["clean_result"] = {
                "prompt": prompt,
                "cleaned": cleaned,
                "original_tokens": original_tokens,
                "cleaned_tokens": cleaned_tokens,
                "removed": removed,
                "reduction": reduction,
                "status": result["status"],
                "chunks": result["n_chunks"],
            }

if "clean_result" in st.session_state:
    data = st.session_state["clean_result"]
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Original Prompt")
        st.text_area("Original prompt output", value=data["prompt"], height=300, disabled=True, label_visibility="collapsed")
        st.metric("Original tokens", data["original_tokens"])
    with right:
        st.subheader("Cleaned Prompt")
        st.text_area("Cleaned prompt", value=data["cleaned"], height=300, key="cleaned_prompt_output", label_visibility="collapsed")
        st.caption("The cleaned prompt is editable and can be copied from the text area.")
        st.metric("Cleaned tokens", data["cleaned_tokens"])

    st.subheader("Compression Statistics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tokens removed", data["removed"])
    m2.metric("Token reduction", f"{data['reduction']:.2f}%")
    m3.metric("Scoring status", data["status"])
    st.caption(f"Model-scoring chunks: {data['chunks']} · Token counts use the notebook scoring tokenizer without special tokens.")
