# 👁️ AI Twin Retina Eyes
Bio-inspired stereo vision with two webcams + two servos (Arduino) and a retina-like pipeline (light adapt → DoG ON/OFF → motion → foveation). Two eyes converge (vergence) to center a target and estimate depth `Z ≈ B / (tanθL + tanθR)`.

This build exposes **HUD tuning** (no training) and emits both:
- a **compact feature vector** (retina L/R pooled + θL/θR + Z + disparity + foveal details) — 24 dims,
- **retinotopic mini-maps** (ON/OFF/MOTION per eye) for visualization / future big-brain input.

Quick start:
```
pip install -r requirements.txt
python -m pc.demo_hud
```
Set camera IDs and serial port in `pc/config.py` if needed.
Keys: `C` sample color • `K` good (+1) • `J` bad (−1) • `Q` quit
