import time, serial
from .config import SERVO
class DualServo:
    def __init__(self, port=None, baud=None):
        self.port=port or SERVO.port; self.baud=baud or SERVO.baud
        self.ser=serial.Serial(self.port, self.baud, timeout=0.2); time.sleep(2.0)
        self.set_left(SERVO.init_left); self.set_right(SERVO.init_right)
    def _clamp(self,deg): return max(SERVO.min_deg, min(SERVO.max_deg, int(deg)))
    def set_left(self,deg): self.ser.write(f"L{self._clamp(deg)}
".encode())
    def set_right(self,deg): self.ser.write(f"R{self._clamp(deg)}
".encode())
    def close(self): 
        try: self.ser.close()
        except: pass
