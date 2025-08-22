import cv2, time, csv, os, numpy as np
from .config import CAM, SERVO, HUD, STEREO
from .webcam import open_cams, read_gray_resized
from .color_tracker import ColorTracker
from .arduino import DualServo
from .vision_features import RetinaStereoPack

LEFT_CAM_ID=CAM.left_id; RIGHT_CAM_ID=CAM.right_id; SERIAL_PORT=SERVO.port

class Tuner:
    def __init__(self, win="Tuning"):
        self.win=win
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL); cv2.resizeWindow(self.win, 360, 260)
        cv2.createTrackbar("k_x100", self.win, 150, 400, lambda v: None)
        cv2.createTrackbar("DoG_small(odd)", self.win, 1, 7, lambda v: None)
        cv2.createTrackbar("DoG_big(odd)", self.win, 4, 16, lambda v: None)
        cv2.createTrackbar("ring1_%", self.win, 20, 40, lambda v: None)
        cv2.createTrackbar("ring2_%", self.win, 50, 80, lambda v: None)
    def params(self):
        k=max(1,cv2.getTrackbarPos("k_x100", self.win))/100.0
        s_raw=max(1,cv2.getTrackbarPos("DoG_small(odd)", self.win))
        b_raw=max(1,cv2.getTrackbarPos("DoG_big(odd)", self.win))
        small=2*s_raw+1; big=2*b_raw+1
        r1p=max(5,cv2.getTrackbarPos("ring1_%", self.win))
        r2p=max(r1p+5,cv2.getTrackbarPos("ring2_%", self.win))
        r1,r2=r1p/100.0, r2p/100.0
        return {"k":k,"small":small,"big":big,"r1":r1,"r2":r2}

def draw_hud(frame, cx, cy, ang, label):
    h,w=frame.shape[:2]
    cv2.line(frame,(w//2,0),(w//2,h),(255,255,255),1)
    cv2.line(frame,(0,h//2),(w,h//2),(255,255,255),1)
    if cx is not None and cy is not None:
        cv2.circle(frame,(cx,cy),6,(0,255,0),2)
        sz=32; x1=max(0,cx-sz); y1=max(0,cy-sz); x2=min(w-1,cx+sz); y2=min(h-1,cy+sz)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),1)
    cv2.putText(frame, f"{label} ang={ang:5.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

def thumb3(on, off, mot):
    H,W=on.shape[:2]
    on3=cv2.cvtColor(on, cv2.COLOR_GRAY2BGR)
    off3=cv2.cvtColor(off, cv2.COLOR_GRAY2BGR)
    mot3=cv2.cvtColor(mot, cv2.COLOR_GRAY2BGR)
    row=np.hstack([on3,off3,mot3])
    return cv2.resize(row, (W*3, H), interpolation=cv2.INTER_NEAREST)

def main():
    os.makedirs(os.path.dirname(HUD.csv_path) or ".", exist_ok=True)
    L,R=open_cams(LEFT_CAM_ID, RIGHT_CAM_ID)
    trackerL=ColorTracker(); trackerR=ColorTracker()
    servo=DualServo(SERIAL_PORT, SERVO.baud)
    packer=RetinaStereoPack(); tuner=Tuner()

    angL=SERVO.init_left; angR=SERVO.init_right
    writer=None
    if HUD.record_csv:
        f=open(HUD.csv_path,"w",newline=""); writer=csv.writer(f)
        writer.writerow(["t","thetaL","thetaR","Z","disparity","fovealL","fovealR","k_or_j","features..."])

    print("[HUD] C=center-sample color • K=good(+1) • J=bad(-1) • Q=quit")
    k_or_j=0; prev_errL, prev_errR=0,0

    while True:
        frameL, smallL, _=read_gray_resized(L)
        frameR, smallR, _=read_gray_resized(R)
        if frameL is None or frameR is None: print("Camera read failed."); break

        params=tuner.params()
        maskL=trackerL.mask(frameL); maskR=trackerR.mask(frameR)
        cL=trackerL.centroid(maskL) if maskL is not None else None
        cR=trackerR.centroid(maskR) if maskR is not None else None

        def adjust(centroid, ang, w, prev_err, kp=0.08, kd=0.02):
            if centroid is None: return ang, prev_err
            cx,cy=centroid; err=(w//2 - cx); derr=err - prev_err
            delta=kp*err + kd*derr; return ang+delta, err

        angL, prev_errL = adjust(cL, angL, frameL.shape[1], prev_errL)
        angR, prev_errR = adjust(cR, angR, frameR.shape[1], prev_errR)
        servo.set_left(angL); servo.set_right(angR)

        thetaL=float(angL); thetaR=float(angR)
        vec, maps = packer.pack(
            smallL, smallR, cL, cR, thetaL, thetaR,
            fullL=cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY),
            fullR=cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY),
            params=params,
            frame_size=(frameL.shape[1], frameL.shape[0]),
            return_maps=True
        )
        # Unpack tail of vector
        Z = vec[-5]; disparity = vec[-4]; fovealL = vec[-3]; fovealR = vec[-2] if False else vec[-1]  # see below
        # Correct unpacking: featsL(9)+featsR(9)+[thetaL,thetaR,Z,disp,fovL,fovR] -> 9+9+6=24, indexes: [-6..-1]
        Z, disparity, fovealL, fovealR = vec[-4-1], vec[-3-1], vec[-2-1], vec[-1]

        # HUD overlays
        cxL,cyL=(cL if cL else (None,None)); cxR,cyR=(cR if cR else (None,None))
        draw_hud(frameL, cxL, cyL, thetaL, "Left"); draw_hud(frameR, cxR, cyR, thetaR, "Right")
        if maskL is not None: frameL=cv2.addWeighted(frameL,0.8, cv2.cvtColor(maskL,cv2.COLOR_GRAY2BGR),0.4,0)
        if maskR is not None: frameR=cv2.addWeighted(frameR,0.8, cv2.cvtColor(maskR,cv2.COLOR_GRAY2BGR),0.4,0)

        thumbL=thumb3(maps['L']['on'], maps['L']['off'], maps['L']['motion'])
        thumbR=thumb3(maps['R']['on'], maps['R']['off'], maps['R']['motion'])
        top=np.hstack([frameL, frameR]); bottom=np.hstack([thumbL, thumbR])
        h1,w1=top.shape[:2]; h2,w2=bottom.shape[:2]
        if w2<w1: bottom=np.hstack([bottom, np.zeros((h2,w1-w2,3), dtype=bottom.dtype)])
        elif w1<w2: top=np.hstack([top, np.zeros((h1,w2-w1,3), dtype=top.dtype)])
        both=np.vstack([top,bottom])

        cv2.putText(both, f"Z~{Z:.2f} m  disp={disparity:+.3f}  foveal[L,R]=[{fovealL:.1f}, {fovealR:.1f}]  K/J={k_or_j}",
                    (10, both.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
        cv2.imshow("AI Twin Retina Eyes - HUD", both)

        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),ord('Q')): break
        elif key in (ord('c'),ord('C')): trackerL.sample_center(frameL); trackerR.sample_center(frameR)
        elif key in (ord('k'),ord('K')): k_or_j=+1
        elif key in (ord('j'),ord('J')): k_or_j=-1
        else: k_or_j=0

        if writer: writer.writerow([time.time(), thetaL, thetaR, Z, disparity, fovealL, fovealR, k_or_j] + vec)

    if writer: f.close()
    servo.close(); cv2.destroyAllWindows()

if __name__=="__main__": main()
