/*
 * ============================================================================
 *  FITNESS BAND  —  RAW DATA LOGGER  (for the ML pipeline)
 * ============================================================================
 *
 *  Board:    Seeed XIAO nRF52840
 *  Sensors:  MAX30102 PPG (0x57) + LSM6DS3 IMU (0x6B), same wiring as
 *            fitness_band.ino (SDA -> D4, SCL -> D5, 3V3, GND)
 *
 *  What this sketch does:
 *    Streams every raw sensor reading over USB as CSV lines, 100 times per
 *    second, so ml/record.py can save them for training. It does NOT
 *    compute heart rate or steps; that is the job of the ML pipeline.
 *
 *  Signals logged:
 *    - PPG red + PPG infrared   (the pipeline uses infrared)
 *    - accelerometer x/y/z      (g)
 *    - gyroscope x/y/z          (degrees/second)
 *
 *  Output format, one line per sample:
 *      t_ms,red,ir,ax,ay,az,gx,gy,gz
 *  Lines starting with '#' are messages, not data.
 * ============================================================================
 */

#include <Adafruit_TinyUSB.h>
#include <Wire.h>


// ============================================================================
//  SECTION 1: SETTINGS
// ============================================================================

const uint8_t  PPG_ADDR = 0x57;      // MAX30102
const uint8_t  IMU_ADDR = 0x6B;      // LSM6DS3

const uint8_t  RED_LED_CURRENT = 0x1F;   // ≈ 6 mA. Raise (max 0xFF) if the signal is weak
const uint8_t  IR_LED_CURRENT  = 0x1F;

const uint16_t SAMPLE_PERIOD_MS = 10;    // 100 samples per second

const float ACCEL_G_PER_LSB  = 0.000122; // ±4 g range
const float GYRO_DPS_PER_LSB = 0.0175;   // ±500 dps range


// ============================================================================
//  SECTION 2: I2C HELPERS
// ============================================================================

void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

bool readRegisters(uint8_t addr, uint8_t reg, uint8_t* buf, uint8_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, len) != len) return false;
  for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

uint8_t readRegister(uint8_t addr, uint8_t reg) {
  uint8_t value = 0;
  readRegisters(addr, reg, &value, 1);
  return value;
}


// ============================================================================
//  SECTION 3: IMU (LSM6DS3)
// ============================================================================

float ax, ay, az, gx, gy, gz;   // Latest IMU reading

void setupImu() {
  uint8_t whoAmI = readRegister(IMU_ADDR, 0x0F);
  if (whoAmI != 0x69 && whoAmI != 0x6A) {
    Serial.print("# WARNING: IMU not found, WHO_AM_I = 0x");
    Serial.println(whoAmI, HEX);
  }
  writeRegister(IMU_ADDR, 0x10, 0x48);  // CTRL1_XL: accel 104 Hz, ±4 g
  writeRegister(IMU_ADDR, 0x11, 0x44);  // CTRL2_G:  gyro  104 Hz, ±500 dps
  writeRegister(IMU_ADDR, 0x12, 0x44);  // CTRL3_C:  block data update + auto-increment
}

void readImu() {
  uint8_t buf[12];
  if (!readRegisters(IMU_ADDR, 0x22, buf, 12)) return;   // gyro 0x22-0x27, accel 0x28-0x2D

  int16_t raw[6];
  for (uint8_t i = 0; i < 6; i++) raw[i] = (int16_t)(buf[2 * i] | (buf[2 * i + 1] << 8));

  gx = raw[0] * GYRO_DPS_PER_LSB;
  gy = raw[1] * GYRO_DPS_PER_LSB;
  gz = raw[2] * GYRO_DPS_PER_LSB;
  ax = raw[3] * ACCEL_G_PER_LSB;
  ay = raw[4] * ACCEL_G_PER_LSB;
  az = raw[5] * ACCEL_G_PER_LSB;
}


// ============================================================================
//  SECTION 4: PPG (MAX30102) — RED + INFRARED
//
//  "SpO2 mode" turns on both LEDs. Each sample in the FIFO is 6 bytes:
//  3 bytes red, then 3 bytes infrared.
// ============================================================================

void setupPpg() {
  uint8_t partId = readRegister(PPG_ADDR, 0xFF);
  if (partId != 0x15) {
    Serial.print("# WARNING: MAX30102 not found, PART_ID = 0x");
    Serial.println(partId, HEX);
  }

  writeRegister(PPG_ADDR, 0x09, 0x40);  // Reset
  delay(100);

  writeRegister(PPG_ADDR, 0x08, 0x50);  // FIFO: average 4 samples, overwrite when full
  writeRegister(PPG_ADDR, 0x0A, 0x2F);  // 4096 nA range, 400 samples/s (÷4 = 100/s), 18-bit
  writeRegister(PPG_ADDR, 0x0C, RED_LED_CURRENT);
  writeRegister(PPG_ADDR, 0x0D, IR_LED_CURRENT);
  writeRegister(PPG_ADDR, 0x09, 0x03);  // SpO2 mode: red + infrared

  writeRegister(PPG_ADDR, 0x04, 0x00);  // Clear FIFO pointers
  writeRegister(PPG_ADDR, 0x05, 0x00);
  writeRegister(PPG_ADDR, 0x06, 0x00);
}


// ============================================================================
//  SECTION 5: STREAMING
//
//  Every PPG sample in the FIFO becomes one CSV line, tagged with the
//  latest IMU reading. The sample counter (not millis())
//  sets the timestamp, so samples are exactly 10 ms apart.
// ============================================================================

uint32_t sampleCount = 0;

void printSample(uint32_t red, uint32_t ir) {
  Serial.print(sampleCount * SAMPLE_PERIOD_MS); Serial.print(',');
  Serial.print(red);  Serial.print(',');
  Serial.print(ir);   Serial.print(',');
  Serial.print(ax, 4); Serial.print(',');
  Serial.print(ay, 4); Serial.print(',');
  Serial.print(az, 4); Serial.print(',');
  Serial.print(gx, 2); Serial.print(',');
  Serial.print(gy, 2); Serial.print(',');
  Serial.println(gz, 2);
  sampleCount++;
}

void streamPpgFifo() {
  uint8_t writePtr  = readRegister(PPG_ADDR, 0x04);
  uint8_t readPtr   = readRegister(PPG_ADDR, 0x06);
  uint8_t available = (writePtr - readPtr) & 0x1F;

  while (available > 0) {
    uint8_t count = min(available, (uint8_t)5);   // 5 samples x 6 bytes = 30 bytes per read
    uint8_t buf[30];
    if (!readRegisters(PPG_ADDR, 0x07, buf, count * 6)) return;

    for (uint8_t i = 0; i < count; i++) {
      uint8_t* s = buf + 6 * i;
      uint32_t red = (((uint32_t)s[0] << 16) | ((uint32_t)s[1] << 8) | s[2]) & 0x3FFFF;
      uint32_t ir  = (((uint32_t)s[3] << 16) | ((uint32_t)s[4] << 8) | s[5]) & 0x3FFFF;
      printSample(red, ir);
    }
    available -= count;
  }
}


// ============================================================================
//  SECTION 6: SETUP + LOOP
// ============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);     // Logger only runs while plugged into the laptop
  Wire.begin();
  Wire.setClock(400000);

  setupImu();
  setupPpg();

  Serial.println("# FitBand data logger ready");
  Serial.println("t_ms,red,ir,ax,ay,az,gx,gy,gz");
}

uint32_t lastImuMs = 0;

void loop() {
  uint32_t now = millis();

  if (now - lastImuMs >= SAMPLE_PERIOD_MS) {
    lastImuMs = now;
    readImu();
  }
  streamPpgFifo();
}
