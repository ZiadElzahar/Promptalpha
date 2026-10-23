from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import streamlit as st

from prompt_scoring import (
    MAX_CONTENT_TOKENS_PER_CHUNK,
    SCORING_MODEL_ID,
    WORD_COMBINE,
    compress_prompt,
    load_model,
    score_prompt,
    token_count,
)

st.set_page_config(page_title="Promptalpha", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    div.stButton > button[kind="secondary"] {
        background-color: #d32f2f;
        color: white;
        border: 1px solid #b71c1c;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #b71c1c;
        color: white;
        border-color: #8e0000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def send_report(subject: str, description: str) -> None:
    """Send a report from the configured alternate mailbox to the main mailbox."""
    smtp_host = os.environ.get("REPORT_SMTP_HOST")
    smtp_port = int(os.environ.get("REPORT_SMTP_PORT", "587"))
    smtp_user = os.environ.get("REPORT_SMTP_USER")
    smtp_password = os.environ.get("REPORT_SMTP_PASSWORD")
    report_to = os.environ.get("REPORT_TO_EMAIL")
    if not all((smtp_host, smtp_user, smtp_password, report_to)):
        raise RuntimeError("Report email is not configured.")

    message = EmailMessage()
    message["Subject"] = f"Prompt Cleaner report: {subject}"
    message["From"] = smtp_user
    message["To"] = report_to
    message.set_content(description)
    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(message)


st.title("Promptalpha")
st.caption("Notebook-faithful LLMLingua-2 word-importance scoring and extractive compression")
st.info(
    "This app reuses the notebook's exact scoring model and workflow. "
    "The notebook does not define a 40% cleaning threshold; its demonstrated retention ratio is 50%, "
    "which is the default below."
)

with st.sidebar:
    st.subheader("Notebook configuration")
    st.code(
        f"model = {SCORING_MODEL_ID}\n"
        f"word_combine = {WORD_COMBINE!r}\n"
        f"max_content_tokens_per_chunk = {MAX_CONTENT_TOKENS_PER_CHUNK}\n"
        "token_count = scoring tokenizer, add_special_tokens=False"
    )
    keep_percent = st.slider("Word retention ratio", min_value=1, max_value=100, value=50, step=1)
    st.caption("Compression keeps the highest-scoring words and preserves original whitespace.")

prompt = st.text_area(
    "Original prompt",
    height=280,
    placeholder="Paste the prompt you want to clean...",
)

st.divider()
st.subheader("Report a problem")
st.caption("Send a report to the configured main account. The prompt is not included unless you add it yourself.")
if "show_report_form" not in st.session_state:
    st.session_state["show_report_form"] = False
if st.button("Report a Problem", type="secondary"):
    st.session_state["show_report_form"] = not st.session_state["show_report_form"]

if st.session_state["show_report_form"]:
    with st.form("report_form", clear_on_submit=True):
        report_subject = st.selectbox(
            "Subject",
            ["Too short prompt", "Cleaning result is incorrect", "Model or loading error", "Other"],
        )
        report_description = st.text_area("Description", height=140, placeholder="Describe what happened...")
        report_submitted = st.form_submit_button("Send Report", type="primary")
    if report_submitted:
        if not report_description.strip():
            st.warning("Add a description before sending the report.")
        else:
            try:
                send_report(report_subject, report_description.strip())
            except Exception:
                st.error("The report could not be sent. The app email settings may not be configured correctly.")
            else:
                st.success("Your report was sent.")

if st.button("Clean Prompt", type="primary", use_container_width=True):
    if not prompt or not prompt.strip():
        st.warning("Enter a non-empty prompt before cleaning.")
    else:
        try:
            with st.spinner("Loading the LLMLingua-2 model and scoring the prompt..."):
                bundle = get_model()
                result = score_prompt(prompt, bundle)
                cleaned = compress_prompt(prompt, result, keep_percent / 100)
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
