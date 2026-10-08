from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import analyze

app = FastAPI(title="MEDIQ API", version="1.0.0",
              description="Educational medical-image decision support. Not a medical device.")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_BYTES = 15 * 1024 * 1024

VALIDATION = {  # illustrative numbers - replace with your own held-out results
    "AUROC": 0.912, "Sensitivity": 0.89, "Specificity": 0.84, "ECE (calibrated)": 0.031,
    "Brier score": 0.087, "Abstention rate": "11%", "OOD AUROC": 0.95, "Pointing game": 0.78,
}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "MEDIQ"}


@app.get("/api/validation")
def validation():
    return {"illustrative": True, "metrics": VALIDATION}


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Please upload an image file (PNG/JPG).")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 15 MB).")
    try:
        result = analyze(raw)
    except Exception:
        raise HTTPException(422, "Could not read that image.")
    result["disclaimer"] = "Educational use only. Not a medical device. Requires review by a qualified professional."
    return result
