//Libraries
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
//Accelerometers
Adafruit_MPU6050 mpu;
Adafruit_MPU6050 mpu2;
//Microphones
int micPin = A2;
int micPin2 = A3;

void setup() {
  //Set baud rate same as python code
  Serial.begin(2000000);
  //Initialize accelerometer
  mpu.begin();
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu2.begin();
  mpu2.setAccelerometerRange(MPU6050_RANGE_8_G);
  //Initialize microphone
  pinMode(micPin, INPUT);
  pinMode(micPin2, INPUT);
}
void loop() {
  //Obtain accelerometer data
  sensors_event_t a, g, temp;
  sensors_event_t a2, g2, temp2;
  mpu.getEvent(&a, &g, &temp);
  mpu2.getEvent(&a2, &g2, &temp2);
  //Obtain microphone data
  float val = analogRead(micPin) * (5.0 / 1023.0);
  float val2 = analogRead(micPin2) * (5.0 / 1023.0);
  //Print all (Order: Time, Acc1, Acc2, Sound1, Sound2)
  Serial.print(millis() / 1000.0);
  Serial.print(",");
  delay(250);
  Serial.print(a.acceleration.x);
  Serial.print(",");
  Serial.print(a.acceleration.y);
  Serial.print(",");
  Serial.print(a.acceleration.z);
  Serial.print(",");
  delay(250);
  Serial.print(a2.acceleration.x);
  Serial.print(",");
  Serial.print(a2.acceleration.y);
  Serial.print(",");
  Serial.print(a2.acceleration.z);
  Serial.print(",");
  delay(250);
  Serial.print(val);
  Serial.print(",");
  delay(125);
  Serial.println(val2);
  delay(125); //Total = 1 Hz
}