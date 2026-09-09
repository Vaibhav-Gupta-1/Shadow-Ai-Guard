# Shadow-Ai-Guard
Shadow AI Guard: the browser extension that intercepts prompts/paste/submit events in LLM chat UIs (ChatGPT, Claude, Gemini) via JS content scripts, then runs them through a Python backend with a regex engine (for credentials/API keys) and an NER engine (for PII/corporate secrets) before anything gets sent off.
# Shadow AI Guard

Browser extension + backend to detect and prevent employees from leaking sensitive/confidential company data into public LLMs (ChatGPT, Claude, Gemini, etc.).

**Mini Project Proposal — BCS-554**
Team ID: CS_AIML_3B_07 | PSIT Kanpur | Session 2026-27

## How It Works

1. A browser extension (JS content scripts) intercepts prompt/paste/submit events in LLM chat UIs before the data leaves the browser.
2. The intercepted text is sent to a Python backend (FastAPI) for scanning.
3. The backend runs two detection layers:
   - **Regex engine** — flags structured secrets (API keys, credentials, tokens)
   - **NER engine (ML)** — flags PII and corporate-sensitive entities (names, emails, SSNs, financial data, etc.)
4. If sensitive content is detected, the extension blocks/warns before the prompt is submitted.

## Tech Stack

- **Frontend/Extension:** JavaScript, browser content scripts
- **Backend:** Python, FastAPI, JWT Auth
- **ML:** DistilBERT (token classification / NER), HuggingFace Transformers
- **Frontend (if applicable):** React.js, Tailwind CSS
- **Deployment:** AWS/Azure

## Project Structure

```
ShadowAI_Guard/
├── backend/              # FastAPI backend, model serving
├── extension/            # Browser extension source
├── prepare_dataset.py    # Converts ai4privacy/pii-masking-200k into training CSV
├── train_model.py        # Fine-tunes DistilBERT NER model on the prepared dataset
├── .gitignore
└── README.md
```

## Setup (New Team Members)

### 1. Clone the repo
```bash
git clone https://github.com/Vaibhav-Gupta-1/Shadow-Ai-Guard.git
cd Shadow-Ai-Guard
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install datasets pandas transformers seqeval accelerate torch
```

If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build instead of the default (much faster training):
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Check your CUDA version first with `nvidia-smi`, and adjust `cu121` if needed.

Confirm GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Generate the training dataset
The dataset CSV is **not** tracked in git (too large). Regenerate it locally:
```bash
python prepare_dataset.py
```
This downloads `ai4privacy/pii-masking-200k` from HuggingFace and converts it into `pii_dataset.csv` in the project root.

### 5. Train the model
```bash
python train_model.py
```
This fine-tunes `distilbert-base-uncased` on the prepared dataset and saves the trained model to `./shadowai_pii_model`.

⚠️ Training on CPU is very slow (40+ hours). Use a GPU machine if possible.

## Notes

- `.venv/`, `pii_dataset.csv`, and `shadowai_pii_model/` are git-ignored — each teammate generates/installs these locally rather than pulling them from the repo.
- If you train a model and want to share the weights with the team without re-running training, zip the `shadowai_pii_model/` folder and share it directly (Drive/WeTransfer) rather than committing it to git.
