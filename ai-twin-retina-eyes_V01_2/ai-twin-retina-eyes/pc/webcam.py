import cv2, numpy as np
from .config import CAM
def open_cams(left_id=None, right_id=None):
    L=cv2.VideoCapture(CAM.left_id if left_id is None else left_id)
    R=cv2.VideoCapture(CAM.right_id if right_id is None else right_id)
    for cap in (L,R):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return L,R
def read_gray_resized(cap):
    ok,frame=cap.read()
    if not ok or frame is None: return None,None,None
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray, (CAM.retina_w, CAM.retina_h), interpolation=cv2.INTER_AREA)
    focus=cv2.Laplacian(gray, cv2.CV_64F).var()
    return frame, small, float(focus)
