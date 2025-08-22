from dataclasses import dataclass
@dataclass
class CamConfig:
    left_id:int=0; right_id:int=1; width:int=320; height:int=240; retina_w:int=64; retina_h:int=48
@dataclass
class ServoConfig:
    port:str="COM5"; baud:int=115200; min_deg:int=30; max_deg:int=150; init_left:int=90; init_right:int=90
@dataclass
class StereoConfig:
    baseline_m:float=0.12; eps:float=1e-6
@dataclass
class HUDConfig:
    show_retina_bars:bool=True; record_csv:bool=True; csv_path:str="data/session.csv"
CAM=CamConfig(); SERVO=ServoConfig(); STEREO=StereoConfig(); HUD=HUDConfig()
