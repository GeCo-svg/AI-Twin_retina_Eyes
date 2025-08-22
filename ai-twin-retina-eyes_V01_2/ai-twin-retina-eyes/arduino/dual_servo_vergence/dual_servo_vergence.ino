#include <Servo.h>
Servo servoL; Servo servoR;
const int pinL=9; const int pinR=10;
int minDeg=30, maxDeg=150;
int posL=90, posR=90;
void clampWrite(Servo &s, int &v){ if(v<minDeg)v=minDeg; if(v>maxDeg)v=maxDeg; s.write(v); }
void setup(){ Serial.begin(115200); servoL.attach(pinL); servoR.attach(pinR); delay(500);
  clampWrite(servoL,posL); clampWrite(servoR,posR); Serial.println("READY"); }
void loop(){
  if(Serial.available()){
    String line=Serial.readStringUntil('\n'); line.trim();
    if(line.length()>=2){
      char id=line.charAt(0); int val=line.substring(1).toInt();
      if(id=='L'){ posL=val; clampWrite(servoL,posL); }
      if(id=='R'){ posR=val; clampWrite(servoR,posR); }
    }
  }
}
