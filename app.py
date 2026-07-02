"""FastAPI app that serves the AI Text Summarizer UI + a /summarize JSON API.

The model is a T5 (t5-small) fine-tuned for dialogue summarization, loaded
from ./saved_summary_model.  See saved_summary_model/config.json for the
task-specific defaults baked into the model.
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

import os
import re
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.staticfiles import StaticFiles

# ---- Paths (resolved once, relative to this file so cwd doesn't matter) --------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"
INDEX_HTML = BASE_DIR / "Index.html"

# ---- FastAPI app ---------------------------------------------------------------

app = FastAPI(
    title="Text Summarization API",
    description="An API for text summarization using a fine-tuned T5 model.",
    version="1.0.0",
)

# ---- Model + tokenizer ---------------------------------------------------------
# Lazy-load the first time the module is imported.  Wrapped in a function so we
# only pay the load cost once and so import-time errors give a clear message.

def _load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Make sure ./saved_summary_model exists."
        )
    tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))
    model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))

    # Pick the best available device (Apple MPS -> CUDA -> CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()  # disable dropout for inference
    return model, tokenizer, device


print(f"[startup] Loading model from {MODEL_DIR} ...")
model, tokenizer, device = _load_model()
print(f"[startup] Model ready on device: {device}")

# ---- Schemas -------------------------------------------------------------------

class DialogueInput(BaseModel):
    """Request body for POST /summarize."""
    dialogue: str = Field(..., min_length=1, description="The text to summarize.")


class SummaryOutput(BaseModel):
    """Response body for POST /summarize."""
    summary: str


# ---- Helpers -------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace in raw dialogue/text before tokenization."""
    if not text:
        return ""
    text = re.sub(r"\r\n", " ", text)   # collapse Windows line endings
    text = re.sub(r"\n", " ", text)     # collapse remaining newlines
    text = re.sub(r"\s+", " ", text)    # collapse runs of whitespace
    return text.strip()


def summarize(text: str,
              max_length: int = 200,
              min_length: int = 30,
              length_penalty: float = 2.0,
              num_beams: int = 4) -> str:
    """Run the T5 model on `text` and return the generated summary.

    Defaults are taken from saved_summary_model/config.json ->
    task_specific_params.summarization.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # T5 expects a task prefix.  The model's config specifies "summarize: ".
    input_text = "summarize: " + cleaned

    # Tokenize (max input length = 512, matches the model's n_positions).
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    # Generate without tracking gradients.  use_cache=True speeds up beam search.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,

            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
            use_cache=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()


# ---- Routes --------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the static HTML UI."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Index.html not found")
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health")
async def health():
    """Liveness probe — useful for confirming the model loaded."""
    return {"status": "ok", "device": str(device)}


@app.post("/summarize", response_model=SummaryOutput)
async def summarize_api(payload: DialogueInput):
    """JSON API: takes {'dialogue': '...'} and returns {'summary': '...'}."""
    try:
        summary = summarize(payload.dialogue)
    except Exception as exc:  # surface a clean 500 instead of a stack trace
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {exc}",
        ) from exc
    return SummaryOutput(summary=summary)


# ---- Dev entrypoint ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # `reload=True` needs the `watchfiles` package.  If it's missing we just
    # fall back to a non-reloading server — dev convenience, not required.
    try:
        import watchfiles  # noqa: F401
        reload = True
    except ImportError:
        print("[startup] watchfiles not installed; running without auto-reload.")
        reload = False
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=reload)
