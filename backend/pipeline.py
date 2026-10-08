"""MEDIQ pipeline: quality gate -> model -> calibration -> explainability -> triage.

The default model is a transparent, deterministic PLACEHOLDER built from image
statistics so the whole stack runs with no GPU or weights. Swap `Predictor.logit`
for a real network (e.g. TorchXRayVision DenseNet) -- see README.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

TEMPERATURE = 1.6          # temperature scaling (fit on a validation set in real use)
N_PASSES = 12              # stochastic passes for uncertainty (MC-dropout stand-in)
ABSTAIN_BAND = (0.35, 0.65)
ESCALATE_P = 0.75


# ---------------------------------------------------------------- quality gate
@dataclass
class Quality:
    score: float
    status: str            # PASS | WARN | REJECT
    issues: list[str]
    ood: bool


def load_image(raw: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img.convert("L")


def quality_gate(img: Image.Image) -> Quality:
    a = np.asarray(img, dtype=np.float32) / 255.0
    issues: list[str] = []
    h, w = a.shape

    if min(h, w) < 128:
        issues.append("Resolution too low (<128px)")

    edges = np.asarray(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharp = float(edges.var())
    blur_score = min(1.0, sharp / 400.0)
    if blur_score < 0.15:
        issues.append("Image appears blurred")

    mean, std = float(a.mean()), float(a.std())
    if mean < 0.15:
        issues.append("Underexposed")
    if mean > 0.85:
        issues.append("Overexposed")
    if std < 0.08:
        issues.append("Very low contrast")

    # crude out-of-distribution check: radiographs are low-colour, wide histogram,
    # roughly portrait/square, dark borders. Flag flat/binary/extreme-aspect images.
    hist, _ = np.histogram(a, bins=32, range=(0, 1))
    occupied = float((hist > 0.001 * a.size).sum()) / 32.0
    aspect = w / h
    ood = occupied < 0.25 or not (0.6 <= aspect <= 1.6)
    if ood:
        issues.append("Does not resemble a medical image (OOD)")

    exposure_ok = 1.0 - abs(mean - 0.5) * 1.6
    score = float(np.clip(0.4 * blur_score + 0.3 * max(exposure_ok, 0) + 0.3 * min(std / 0.25, 1), 0, 1))
    if ood or min(h, w) < 128 or score < 0.25:
        status = "REJECT"
    elif issues or score < 0.55:
        status = "WARN"
    else:
        status = "PASS"
    return Quality(round(score, 2), status, issues, ood)


# ------------------------------------------------------------------- predictor
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Predictor:
    """Placeholder scoring head. Replace `logit` with a real model's output."""

    def logit(self, a: np.ndarray, rng: np.random.Generator | None = None) -> float:
        h, w = a.shape
        lower = a[h // 2:, :]
        upper = a[: h // 2, :]
        left, right = lower[:, : w // 2], lower[:, w // 2:]
        z = 6.0 * (lower.mean() - upper.mean()) + 3.0 * (right.mean() - left.mean()) - 0.2
        if rng is not None:                       # dropout-style perturbation
            keep = rng.random(a.shape) > 0.1
            z += float(rng.normal(0, 0.25)) + 0.5 * float((a * keep).mean() - a.mean())
        return float(z)


def calibrated_prob(logit: float) -> float:
    return _sigmoid(logit / TEMPERATURE)


def uncertainty(pred: Predictor, a: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    ps = np.array([calibrated_prob(pred.logit(a, rng)) for _ in range(N_PASSES)])
    mean = float(ps.mean())
    lo, hi = np.percentile(ps, [5, 95])
    return mean, float(lo), float(hi), float(ps.std())


# ------------------------------------------------------------- explainability
def attention_map(a: np.ndarray, grid: int = 8) -> list[list[float]]:
    """Coarse saliency grid (occlusion-style) normalised to 0..1."""
    h, w = a.shape
    pred = Predictor()
    base = pred.logit(a)
    out = np.zeros((grid, grid), dtype=np.float32)
    ys, xs = np.linspace(0, h, grid + 1, dtype=int), np.linspace(0, w, grid + 1, dtype=int)
    for i in range(grid):
        for j in range(grid):
            b = a.copy()
            b[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = a.mean()
            out[i, j] = abs(base - pred.logit(b))
    if out.max() > 0:
        out /= out.max()
    return np.round(out, 3).tolist()


def evidence(a: np.ndarray, att: list[list[float]], p: float) -> list[str]:
    g = np.array(att)
    i, j = np.unravel_index(g.argmax(), g.shape)
    n = g.shape[0]
    vert = "upper" if i < n / 3 else "mid" if i < 2 * n / 3 else "lower"
    horiz = "left" if j < n / 2 else "right"
    out = [f"Model attention concentrated in the {horiz} {vert} zone"]
    out.append("Attention is focal (a few regions dominate)" if (g > 0.6).sum() <= 6
               else "Attention is diffuse across the image")
    out.append("Score above the escalation threshold" if p >= ESCALATE_P
               else "Score within the abstention band" if ABSTAIN_BAND[0] <= p <= ABSTAIN_BAND[1]
               else "Score below the positive threshold")
    return out


# ---------------------------------------------------------------------- triage
def triage(p: float, spread: float, q: Quality) -> tuple[str, str]:
    if q.status == "REJECT":
        return "REJECT", "Image failed the quality gate - re-acquire or upload a suitable image."
    if spread > 0.15 or ABSTAIN_BAND[0] <= p <= ABSTAIN_BAND[1]:
        return "DEFER", "Model is uncertain - abstaining. Refer for qualified human review."
    if p >= ESCALATE_P:
        return "ESCALATE", "Priority review by a qualified clinician."
    if q.status == "WARN":
        return "DEFER", "Borderline image quality - confirm with a clinician before relying on this."
    return "ROUTINE", "Low-risk output - routine review by a qualified clinician."


def analyze(raw: bytes) -> dict:
    img = load_image(raw)
    q = quality_gate(img)
    small = img.resize((256, 256))
    a = np.asarray(small, dtype=np.float32) / 255.0

    result = {
        "quality": {"score": q.score, "status": q.status, "issues": q.issues, "ood": q.ood},
    }
    if q.status == "REJECT":
        action, msg = triage(0.5, 1.0, q)
        result.update(finding="No prediction made", label="rejected", probability=None,
                      interval=None, evidence=q.issues, attention=None,
                      triage={"action": action, "message": msg})
        return result

    p, lo, hi, spread = uncertainty(Predictor(), a)
    att = attention_map(a)
    action, msg = triage(p, spread, q)
    label = "positive" if p >= 0.5 else "negative"
    if action == "DEFER" and ABSTAIN_BAND[0] <= p <= ABSTAIN_BAND[1]:
        label = "uncertain"
    result.update(
        finding={"positive": "Possible abnormality flagged", "negative": "No abnormality flagged",
                 "uncertain": "Uncertain - model abstains"}[label],
        label=label, probability=round(p, 3), interval=[round(lo, 3), round(hi, 3)],
        evidence=evidence(a, att, p), attention=att,
        triage={"action": action, "message": msg},
    )
    return result
