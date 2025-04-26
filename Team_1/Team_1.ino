//Accelerometer
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

//Microphone
int micPin = A3;

//Sampling rate
#define PERIOD 1
unsigned long time = 0L;

void setup() {
  //Set baud rate
  Serial.begin(9600); //same as python code

  //Initialize accelerometer and microphone
  mpu.begin();
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
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
  if (millis() - time > PERIOD) {
    time += PERIOD;
    sample();
  }
}

void sample() {

  //Obtain accelerometer data
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float val = analogRead(micPin) * (5.0 / 1023.0) ;

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
