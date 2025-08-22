import cv2
import numpy as np
import argparse
import serial

def retina_block(frame, prev_frame=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.log1p(0.1 * gray.astype(np.float32))
    blur_small = cv2.GaussianBlur(gray, (3,3), 0)
    blur_big = cv2.GaussianBlur(gray, (9,9), 0)
    ON = np.clip(blur_small - blur_big, 0, None)
    OFF = np.clip(blur_big - blur_small, 0, None)
    MOTION = np.zeros_like(gray) if prev_frame is None else np.clip(gray - prev_frame, 0, None)
    return ON, OFF, MOTION, gray

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=int, default=0, help="Index cam left")
    parser.add_argument("--right", type=int, default=1, help="Index cam right")
    parser.add_argument("--serial", type=str, default=None, help="Serial port Arduino (optional)")
    parser.add_argument("--baseline", type=float, default=0.1, help="Distance between eyes (m)")
    args = parser.parse_args()

    capL, capR = cv2.VideoCapture(args.left), cv2.VideoCapture(args.right)
    ser = serial.Serial(args.serial, 9600) if args.serial else None

    prevL, prevR = None, None
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        ONL, OFFL, MOTL, prevL = retina_block(frameL, prevL)
        ONR, OFFR, MOTR, prevR = retina_block(frameR, prevR)

        disp = cv2.absdiff(prevL, prevR)
        disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        top = np.hstack([frameL, frameR])
        bottom = np.hstack([cv2.convertScaleAbs(ONL), cv2.convertScaleAbs(OFFL), depth])
        stacked = np.vstack([top, bottom])

        cv2.imshow("HUD Two Eyes Vision", stacked)

        if ser:
            ser.write(b'X')  # esempio banale
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release(), capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
