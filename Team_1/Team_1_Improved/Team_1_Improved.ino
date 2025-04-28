//Libraries
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

//Accelerometers
Adafruit_MPU6050 mpu;

//Microphones
int micPin = A0;

void setup() {
  //Set baud rate same as python code
  Serial.begin(9600);

  //Initialize accelerometer
  mpu.begin();
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);

  //Initialize microphone
  pinMode(micPin, INPUT);

  //Print headers
  Serial.print("Time");
  Serial.print(",");
  Serial.print("X");
  Serial.print(",");
  Serial.print("Y");
  Serial.print(",");
  Serial.print("Z");
  Serial.print(",");
  Serial.println("Sound");
  
}

void loop() {
  //Obtain accelerometer data
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  //Obtain microphone data
  float val = analogRead(micPin) * (5.0 / 1023.0);

  //Print all
  Serial.print(millis());
  Serial.print(",");
  Serial.print(a.acceleration.x);
  Serial.print(",");
  Serial.print(a.acceleration.y);
  Serial.print(",");
  Serial.print(a.acceleration.z);
  Serial.print(",");
  Serial.println(val);
}