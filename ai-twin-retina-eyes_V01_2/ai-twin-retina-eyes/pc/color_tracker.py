import cv2, numpy as np
class ColorTracker:
    def __init__(self): self.hsv_sample=None; self.tol=(10,60,60)
    def sample_center(self, frame_bgr):
        h,w,_=frame_bgr.shape; cx,cy=w//2,h//2
        patch=frame_bgr[cy-5:cy+5, cx-5:cx+5]
        hsv=cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mean=hsv.reshape(-1,3).mean(axis=0).astype(np.uint8)
        self.hsv_sample=tuple(int(x) for x in mean)
    def mask(self, frame_bgr):
        if self.hsv_sample is None: return None
        hsv=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h,s,v=self.hsv_sample; dh,ds,dv=self.tol
        lower=np.array([max(0,h-dh), max(0,s-ds), max(0,v-dv)], dtype=np.uint8)
        upper=np.array([min(179,h+dh), min(255,s+ds), min(255,v+dv)], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)
    def centroid(self, mask):
        if mask is None: return None
        M=cv2.moments(mask)
        if M["m00"]>0: return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        return None
