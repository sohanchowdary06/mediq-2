# MEDIQ — Clinical Evidence & Triage Support

Educational medical-image decision support: quality gate, calibrated confidence,
visible evidence and safe triage. **Not a medical device. No diagnosis.**
Use only public, de-identified images.

## Run

```bash
# backend (http://localhost:8000, docs at /docs)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# frontend (separate terminal) -> http://localhost:5173
cd frontend
python -m http.server 5173
```

The frontend calls `http://localhost:8000` by default. Change `API_BASE` at the
top of `frontend/app.js` for deployment.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze` | multipart `file` -> label, probability, interval, evidence, attention grid, triage |
| GET | `/api/validation` | illustrative validation metrics |
| GET | `/api/health` | health check |

## Important: the model is a placeholder

`backend/pipeline.py` ships with a transparent, image-statistics scoring head so
everything runs without weights. Its outputs are **not clinically meaningful**.
To make it real, replace `Predictor.logit` with a pretrained network
(e.g. TorchXRayVision DenseNet), fit `TEMPERATURE` on a validation set, and
replace `attention_map` with Grad-CAM. The quality gate, uncertainty,
abstention and triage logic stay the same. The validation numbers shown in the
UI are illustrative and must be replaced with your own.
