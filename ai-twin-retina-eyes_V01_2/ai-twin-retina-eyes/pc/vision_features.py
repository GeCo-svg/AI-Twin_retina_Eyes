import cv2, numpy as np
from .retina import retina_maps_and_features, MotionHP
from .stereo import depth_from_vergence

def _norm_disparity(cL, cR, frame_w):
    if (cL is None) or (cR is None): return 0.0
    dx=float(cR[0]-cL[0]); return dx/max(1.0,float(frame_w))

def _foveal_detail(gray_full, c, box=64):
    if c is None or gray_full is None: return 0.0
    h,w=gray_full.shape[:2]; cx,cy=c
    x1=int(np.clip(cx-box//2,0,w-1)); y1=int(np.clip(cy-box//2,0,h-1))
    x2=int(np.clip(x1+box,0,w)); y2=int(np.clip(y1+box,0,h))
    crop=gray_full[y1:y2, x1:x2]
    if crop.size==0: return 0.0
    fm=cv2.Laplacian(crop, cv2.CV_64F).var(); return float(fm)

class RetinaStereoPack:
    def __init__(self):
        self.mL=MotionHP(); self.mR=MotionHP()
    def pack(self, grayL_small, grayR_small, centerL, centerR,
             thetaL, thetaR, fullL=None, fullR=None, params=None,
             frame_size=None, return_maps=False):
        mapsL, featsL = retina_maps_and_features(grayL_small, center=centerL, params=params, motion_state=self.mL)
        mapsR, featsR = retina_maps_and_features(grayR_small, center=centerR, params=params, motion_state=self.mR)
        Z = depth_from_vergence(thetaL, thetaR)
        frame_w = frame_size[0] if frame_size else grayL_small.shape[1]*5
        disp = _norm_disparity(centerL, centerR, frame_w=frame_w)
        fovL = _foveal_detail(fullL, centerL, box=64)
        fovR = _foveal_detail(fullR, centerR, box=64)
        vec = featsL + featsR + [float(thetaL), float(thetaR), Z, disp, fovL, fovR]
        if not return_maps: return vec, None
        def to_u8(m): return (np.clip(m,0.0,1.0)*255.0).astype(np.uint8)
        maps_opt={'L':{k:to_u8(v) for k,v in mapsL.items()}, 'R':{k:to_u8(v) for k,v in mapsR.items()}}
        return vec, maps_opt
