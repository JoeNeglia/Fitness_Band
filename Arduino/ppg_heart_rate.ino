#include <Adafruit_TinyUSB.h>
#include <Wire.h>

#define MAX_ADDR 0x57

void writeRegister(byte reg, byte value) {
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  writeRegister(0x09, 0x40);
  delay(100);
  writeRegister(0x08, 0x00);
  writeRegister(0x09, 0x02);
  writeRegister(0x0A, 0x27);
  writeRegister(0x0C, 0x1F);
  writeRegister(0x04, 0x00);
  writeRegister(0x05, 0x00);
  writeRegister(0x06, 0x00);

  // Print CSV header
  Serial.println("timestamp_ms,raw,filtered");
}

float filtered = 0;
const float alpha = 0.08;

void loop() {
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(0x07);
  Wire.endTransmission(false);
  Wire.requestFrom(MAX_ADDR, 3);

  if (Wire.available() >= 3) {
    uint32_t sample = 0;
    sample |= ((uint32_t)Wire.read() << 16);
    sample |= ((uint32_t)Wire.read() << 8);
    sample |= Wire.read();
    sample &= 0x3FFFF;

    filtered = alpha * sample + (1.0 - alpha) * filtered;

    // Print as CSV: timestamp, raw, filtered
    Serial.print(millis());
    Serial.print(",");
    Serial.print(sample);
    Serial.print(",");
    Serial.println((uint32_t)filtered);
  }

  delay(20);
}