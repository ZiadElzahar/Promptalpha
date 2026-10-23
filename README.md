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
- Compression: top-scoring words are retained; original whitespace is preserved and dropped word characters are removed
- Retention ratio used by the app: fixed at `30%`

The user interface uses a fixed 30% retention ratio as requested. The internal scoring implementation remains unchanged.

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

Paste a prompt, choose the word retention ratio if desired, and click **Clean Prompt**. The app displays the original prompt, cleaned prompt, original and cleaned token counts, tokens removed, reduction percentage, and model scoring status.

## Report email configuration

The red **Report a Problem** button opens a form with a subject selector and description field. To send reports from the website/alternate mailbox to your main mailbox, configure SMTP environment variables before starting Streamlit:

```bash
export REPORT_SMTP_HOST="smtp.example.com"
export REPORT_SMTP_PORT="587"
export REPORT_SMTP_USER="your-alternate-account@example.com"
export REPORT_SMTP_PASSWORD="your-smtp-app-password"
export REPORT_TO_EMAIL="your-main-account@example.com"
streamlit run app.py
```

Use an app password or SMTP-specific credential where the mail provider supports it. Do not put the password in `app.py`, commit it to Git, or expose it in the browser. For Gmail, `REPORT_SMTP_HOST` is usually `smtp.gmail.com`, port `587`, and the sending account must have SMTP/app-password access enabled. The app sends the selected subject and description only; it does not automatically attach the user's prompt.

## Validation performed

The implementation was checked for Python syntax and imports of the local modules. The full model inference was not run in this sandbox because it requires downloading the Hugging Face checkpoint and may require substantial resources. The app reports a user-facing processing error rather than fabricating output or statistics if model loading/inference fails.

## Limitations

The notebook's model was trained on meeting transcripts, so applying it to code-generation prompts has documented domain shift. The app performs word-level extractive compression, not generative rewriting. Long prompts are scored in independent chunks, as in the notebook, so cross-chunk context is limited. GPU use is automatic when CUDA is available; otherwise inference runs on CPU and may be slow.
