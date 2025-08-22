import math
from .config import STEREO

def depth_from_vergence(thetaL_deg, thetaR_deg, baseline_m=None):
    B = STEREO.baseline_m if baseline_m is None else baseline_m
    thL = math.radians(thetaL_deg); thR = math.radians(thetaR_deg)
    denom = math.tan(thL) + math.tan(thR)
    if abs(denom) < STEREO.eps: return float("inf")
    Z = B / denom
    return float(abs(Z))
