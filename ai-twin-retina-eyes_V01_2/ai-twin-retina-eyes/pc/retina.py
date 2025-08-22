import numpy as np, cv2

class MotionHP:
    def __init__(self): self.prev=None
    def step(self,y):
        if self.prev is None:
            self.prev=y.copy(); return np.zeros_like(y)
        mot=np.maximum(y-self.prev,0.0); self.prev=y.copy(); return mot

def _odd(k:int)->int:
    k=max(1,int(k)); return k if (k%2)==1 else (k+1)

def light_adapt(x, k=1.5):
    xf=x.astype(np.float32)/255.0
    y=np.log1p(k*xf); y/= (y.max()+1e-6)
    blur=cv2.blur(y,(5,5)); y=y/(blur+1e-3)
    return np.clip(y,0.0,1.0)

def dog_on_off(y, small=3, big=9):
    small=_odd(small); big=_odd(big)
    if big<=small: big=small+2
    s=cv2.GaussianBlur(y,(small,small),0); b=cv2.GaussianBlur(y,(big,big),0)
    on=np.maximum(s-b,0.0); off=np.maximum(b-s,0.0); return on,off

def foveation_pool(m, center=None, rings=(0.2,0.5,1.0)):
    H,W=m.shape; cx,cy=(W//2,H//2) if center is None else center
    cx=int(np.clip(cx,0,W-1)); cy=int(np.clip(cy,0,H-1))
    yy,xx=np.mgrid[0:H,0:W]; rr=np.sqrt((xx-cx)**2+(yy-cy)**2)
    rmax=np.sqrt((W/2)**2+(H/2)**2)+1e-6; r=rr/rmax
    r1,r2,_=rings; r1=np.clip(r1,0.0,1.0); r2=np.clip(r2,r1+1e-3,1.0); r3=1.0
    fov=(r<=r1); mid=(r>r1)&(r<=r2); per=(r>r2)&(r<=r3)
    return [float(m[fov].mean() if fov.any() else 0.0),
            float(m[mid].mean() if mid.any() else 0.0),
            float(m[per].mean() if per.any() else 0.0)]

def retina_maps_and_features(gray_small, center=None, params=None, motion_state=None):
    if params is None: params={}
    k=float(params.get('k',1.5)); small=int(params.get('small',3)); big=int(params.get('big',9))
    r1=float(params.get('r1',0.2)); r2=float(params.get('r2',0.5))
    y=light_adapt(gray_small, k=k)
    on,off=dog_on_off(y, small=small, big=big)
    if motion_state is None: motion_state=MotionHP()
    mot=motion_state.step(y)
    maps={'on':on,'off':off,'motion':mot}
    feats=[]
    for ch in (on,off,mot): feats+=foveation_pool(ch, center=center, rings=(r1,r2,1.0))
    return maps, feats
