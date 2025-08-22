#include <Servo.h>
Servo servoX, servoY;

void setup() {
  servoX.attach(9);
  servoY.attach(10);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'X') {
      servoX.write(90);
      servoY.write(90);
    }
  }
}
