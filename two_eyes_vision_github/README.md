# Two Eyes Vision
Sistema di visione ispirato all'occhio umano con due webcam e servo-occhi controllati da Arduino.

## Features
- Retina artificiale (ON, OFF, MOTION)
- Stereo depth map (differenza disparità)
- HUD PC
- Arduino per pan/tilt occhi

## Uso
```bash
pip install -r requirements.txt
python -m pc.demo_hud --left 0 --right 1 --serial COM5 --baseline 0.12
```
Premi **Q** per uscire.

## Cartelle
- `pc/` codice Python HUD
- `arduino/` sketch Arduino
