// Libraries
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// Accelerometers
Adafruit_MPU6050 mpu;
Adafruit_MPU6050 mpu2;

// Microphones
const int micPin = A2;
const int micPin2 = A3;

// Timing
unsigned long lastSend = 0;
const unsigned long sendInterval = 100; // milliseconds (sampling rate = 10 Hz)

void setup() {
  Serial.begin(2000000);

  // Initialize accelerometers
  mpu.begin();
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu2.begin();
  mpu2.setAccelerometerRange(MPU6050_RANGE_8_G);

  // Initialize microphones
  pinMode(micPin, INPUT);
  pinMode(micPin2, INPUT);
}

void loop() {
  unsigned long now = millis();
  if (now - lastSend >= sendInterval) {
    lastSend = now;

    // Obtain accelerometer data
    sensors_event_t a, g, temp;
    sensors_event_t a2, g2, temp2;
    mpu.getEvent(&a, &g, &temp);
    mpu2.getEvent(&a2, &g2, &temp2);

    // Obtain microphone data
    float val = analogRead(micPin) * (5.0 / 1023.0);
    float val2 = analogRead(micPin2) * (5.0 / 1023.0);

    // Use snprintf for efficient string formatting
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
      a.acceleration.x, a.acceleration.y, a.acceleration.z,
      a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
      val, val2);

    Serial.println(buffer);
  }
}
