/*
 * ============================================================================
 *  FIT BAND BLE  —  streams sensor data to collect.py over Bluetooth
 * ============================================================================
 *
 *  Board:    Seeed XIAO nRF52840
 *            (Arduino IDE: Tools > Board > Seeed nRF52 Boards > Seeed XIAO nRF52840)
 *
 *  Sensors (both on the same I2C bus):
 *            MAX30102 PPG heart-rate sensor  -> I2C address 0x57
 *            LSM6DS3  IMU (accel + gyro)     -> I2C address 0x6B
 *
 *  Wiring:   Sensor SDA -> XIAO D4 (SDA)
 *            Sensor SCL -> XIAO D5 (SCL)
 *            Sensor VIN -> XIAO 3V3
 *            Sensor GND -> XIAO GND
 *
 *  What this sketch does:
 *    1. Reads the PPG sensor 100 times per second and runs a simple
 *       on-board beat detector (the "bpm" column).
 *    2. Reads the IMU 100 times per second to measure motion and count steps.
 *    3. About 10 times per second, sends one CSV text line over Bluetooth
 *       using the Nordic UART Service (NUS). collect.py receives it.
 *       The same line is also printed to the USB Serial Monitor.
 *
 *  CSV line format (13 columns, no header):
 *      ms,ppg,pulse,ax_mg,ay_mg,az_mg,gx_dps,gy_dps,gz_dps,motion,bpm,status,steps
 *    ms       board time in milliseconds since power-on
 *    ppg      raw PPG light level, averaged over the samples since the last line
 *    pulse    on-board filtered pulse wave (flipped so beats are peaks)
 *    ax..az   acceleration in milli-g
 *    gx..gz   rotation speed in degrees/second
 *    motion   on-board motion level: 0 still, 1 moderate, 2 heavy
 *    bpm      on-board heart rate estimate (0 = no reading)
 *    status   0 no skin, 1 searching, 2 ok, 3 too much motion
 *    steps    steps counted since power-on
 *
 *  To stream faster later, lower STREAM_PERIOD_MS (10 ms = 100 lines/s).
 * ============================================================================
 */

#include <Adafruit_TinyUSB.h>   // Needed for Serial on the Seeed nRF52 core
#include <Wire.h>               // I2C communication with the sensors
#include <bluefruit.h>          // Bluetooth Low Energy (built into the Seeed nRF52 core)


// ============================================================================
//  SECTION 1: SETTINGS YOU CAN TUNE
// ============================================================================

// ---- I2C addresses ----
const uint8_t PPG_ADDR = 0x57;  // MAX30102
const uint8_t IMU_ADDR = 0x6B;  // LSM6DS3

// ---- Heart rate ----
// Raw PPG level above this means skin is on the sensor. Check the "ppgDC"
// value in the Serial Plotter with and without your finger/wrist on it,
// then pick a number in between.
const uint32_t SKIN_THRESHOLD  = 30000;
const uint16_t MIN_BEAT_GAP_MS = 300;   // 300 ms between beats = 200 BPM max
const uint16_t MAX_BEAT_GAP_MS = 1500;  // 1500 ms between beats = 40 BPM min
const uint8_t  BEATS_TO_AVERAGE = 8;    // BPM = median of the last 8 beat gaps
const uint16_t HR_TIMEOUT_MS   = 8000;  // No good beat for this long -> show no BPM

// ---- Motion noise filter ----
// "Dynamic acceleration" = how much acceleration there is beyond gravity, in g.
// Gyro values are rotation speed in degrees per second.
const float    MOTION_MODERATE_G   = 0.15;  // Walking-level motion
const float    MOTION_HEAVY_G      = 0.50;  // Running / shaking-level motion
const float    MOTION_MODERATE_DPS = 60.0;
const float    MOTION_HEAVY_DPS    = 200.0;
const uint16_t MOTION_SETTLE_MS    = 1500;  // After heavy motion, wait this long before trusting beats again

// ---- Step counter ----
const float    STEP_THRESHOLD_G     = 0.12;  // Acceleration bump needed to count as a step
const uint16_t STEP_MIN_INTERVAL_MS = 250;   // Steps closer than this are ignored (bounces)
const uint16_t STEP_MAX_INTERVAL_MS = 2000;  // Gaps longer than this end a walking streak
const uint8_t  STEPS_TO_START_WALK  = 4;     // Need this many regular steps in a row before counting

// ---- Bluetooth streaming ----
const char*    DEVICE_NAME      = "FitBand";   // collect.py looks for this name
const uint16_t STREAM_PERIOD_MS = 100;         // 100 ms = 10 lines per second
const bool     ECHO_TO_SERIAL   = true;        // also print each line to the USB Serial Monitor


// ============================================================================
//  SECTION 2: SHARED STATE
//  Values that several sections read or write.
// ============================================================================

// Heart rate status codes (the "status" column).
enum HrStatus : uint8_t {
  HR_NO_SKIN   = 0,   // Sensor isn't touching skin
  HR_SEARCHING = 1,   // Skin detected, still collecting beats
  HR_OK        = 2,   // Reliable heart rate
  HR_MOTION    = 3    // Too much movement, holding the last good value
};

// Motion levels decided by the IMU (Section 5).
enum MotionLevel : uint8_t {
  MOTION_STILL    = 0,
  MOTION_MODERATE = 1,
  MOTION_HEAVY    = 2
};

// One reading from the IMU. (Declared up here because the Arduino IDE
// needs custom types defined before the first function.)
struct ImuReading {
  float ax, ay, az;   // acceleration in g
  float gx, gy, gz;   // rotation in degrees/second
};

uint8_t     currentBpm      = 0;            // 0 = no reading
HrStatus    hrStatus        = HR_NO_SKIN;
MotionLevel motionLevel     = MOTION_STILL;
uint32_t    lastHeavyMotionMs = 0;          // When heavy motion was last seen
uint32_t    stepsToday      = 0;            // steps since power-on
ImuReading  latestImu       = {0, 0, 0, 0, 0, 0};   // newest IMU reading, for streaming


// ============================================================================
//  SECTION 3: I2C HELPER FUNCTIONS
//  Small wrappers so the sensor code can read/write registers in one line.
// ============================================================================

void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

// Reads `len` bytes starting at register `reg`. Returns false on an I2C error.
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
//  SECTION 4: IMU (LSM6DS3) SETUP AND READING
// ============================================================================

// Conversion factors from the LSM6DS3 datasheet for the ranges chosen below.
const float ACCEL_G_PER_LSB   = 0.000122;   // ±4 g range
const float GYRO_DPS_PER_LSB  = 0.0175;     // ±500 dps range

void setupImu() {
  uint8_t whoAmI = readRegister(IMU_ADDR, 0x0F);   // WHO_AM_I: 0x69 (DS3) or 0x6A (DS3TR-C)
  if (whoAmI != 0x69 && whoAmI != 0x6A) {
    Serial.print("WARNING: IMU not found at 0x6B (WHO_AM_I = 0x");
    Serial.print(whoAmI, HEX);
    Serial.println(")");
  }

  writeRegister(IMU_ADDR, 0x10, 0x48);  // CTRL1_XL: accel 104 Hz, ±4 g
  writeRegister(IMU_ADDR, 0x11, 0x44);  // CTRL2_G:  gyro  104 Hz, ±500 dps
  writeRegister(IMU_ADDR, 0x12, 0x44);  // CTRL3_C:  block data update + register auto-increment
}

// Reads gyro + accel in one go (registers 0x22..0x2D are back to back).
bool readImu(ImuReading& r) {
  uint8_t buf[12];
  if (!readRegisters(IMU_ADDR, 0x22, buf, 12)) return false;

  int16_t raw[6];
  for (uint8_t i = 0; i < 6; i++) {
    raw[i] = (int16_t)(buf[2 * i] | (buf[2 * i + 1] << 8));   // low byte first
  }

  r.gx = raw[0] * GYRO_DPS_PER_LSB;
  r.gy = raw[1] * GYRO_DPS_PER_LSB;
  r.gz = raw[2] * GYRO_DPS_PER_LSB;
  r.ax = raw[3] * ACCEL_G_PER_LSB;
  r.ay = raw[4] * ACCEL_G_PER_LSB;
  r.az = raw[5] * ACCEL_G_PER_LSB;
  return true;
}


// ============================================================================
//  SECTION 5: MOTION DETECTION (the motion noise filter)
//
//  Wrist movement pushes blood around and shifts the sensor on the skin,
//  which creates fake "beats" in the PPG signal. This section measures how
//  much the wrist is moving so Section 8 can:
//    - STILL:    accept beats normally
//    - MODERATE: only accept beats that match the recent heart rate closely
//    - HEAVY:    reject all beats, keep showing the last good heart rate
// ============================================================================

float gravityG       = 1.0;   // Slow average of acceleration size (≈ gravity)
float dynamicAccelG  = 0.0;   // Acceleration beyond gravity, right now
float motionAccelAvg = 0.0;   // Smoothed dynamic acceleration
float motionGyroAvg  = 0.0;   // Smoothed rotation speed

void updateMotion(const ImuReading& r, uint32_t nowMs) {
  // Size of the acceleration vector. Holding still, this is about 1.0 g
  // no matter which way the wrist is tilted.
  float accelMag = sqrtf(r.ax * r.ax + r.ay * r.ay + r.az * r.az);
  float gyroMag  = sqrtf(r.gx * r.gx + r.gy * r.gy + r.gz * r.gz);

  // Track gravity slowly (≈1 second), then subtract it to keep only movement.
  gravityG      += 0.01 * (accelMag - gravityG);
  dynamicAccelG  = accelMag - gravityG;

  // Smooth the motion measurements so one bump doesn't flip the level.
  motionAccelAvg += 0.05 * (fabsf(dynamicAccelG) - motionAccelAvg);
  motionGyroAvg  += 0.05 * (gyroMag - motionGyroAvg);

  // Decide the motion level.
  if (motionAccelAvg > MOTION_HEAVY_G || motionGyroAvg > MOTION_HEAVY_DPS) {
    motionLevel = MOTION_HEAVY;
    lastHeavyMotionMs = nowMs;
  } else if (motionAccelAvg > MOTION_MODERATE_G || motionGyroAvg > MOTION_MODERATE_DPS) {
    motionLevel = MOTION_MODERATE;
  } else {
    motionLevel = MOTION_STILL;
  }
}

// True while moving heavily, and for a short time after, while the PPG
// signal settles back down.
bool motionTooHigh(uint32_t nowMs) {
  return motionLevel == MOTION_HEAVY || (nowMs - lastHeavyMotionMs) < MOTION_SETTLE_MS;
}


// ============================================================================
//  SECTION 6: STEP COUNTER
//
//  Each step makes a bump in acceleration. A step is counted when the
//  smoothed dynamic acceleration rises above STEP_THRESHOLD_G.
//
//  Noise filter: random arm movements (typing, waving, scratching) make
//  bumps too, but not in a steady rhythm. So steps are held as "pending"
//  until STEPS_TO_START_WALK regular steps happen in a row. Only then are
//  they added to the total, and later steps count right away. A pause
//  longer than STEP_MAX_INTERVAL_MS ends the walking streak.
// ============================================================================

float    stepSignal     = 0.0;    // Smoothed dynamic acceleration
bool     stepArmed      = true;   // Ready to detect the next step
uint32_t lastStepMs     = 0;
uint8_t  pendingSteps   = 0;      // Steps waiting for a steady rhythm
bool     walking        = false;  // In a steady walking streak

void onStepDetected(uint32_t nowMs) {
  uint32_t gap = nowMs - lastStepMs;

  if (gap < STEP_MIN_INTERVAL_MS) return;   // Too soon: a bounce of the same step

  lastStepMs = nowMs;

  if (gap > STEP_MAX_INTERVAL_MS) {
    // Long pause: start a new streak with this step as pending.
    walking = false;
    pendingSteps = 1;
    return;
  }

  if (walking) {
    stepsToday++;
  } else {
    pendingSteps++;
    if (pendingSteps >= STEPS_TO_START_WALK) {
      walking = true;
      stepsToday += pendingSteps;    // Credit the steps that were held back
      pendingSteps = 0;
    }
  }
}

void updateSteps(uint32_t nowMs) {
  // Light smoothing removes sharp spikes from the step signal.
  stepSignal += 0.25 * (dynamicAccelG - stepSignal);

  // Count once when the signal goes above the threshold, then wait for it
  // to drop back below zero before counting again (prevents double counts).
  if (stepArmed && stepSignal > STEP_THRESHOLD_G) {
    stepArmed = false;
    onStepDetected(nowMs);
  } else if (!stepArmed && stepSignal < 0.0) {
    stepArmed = true;
  }
}


// ============================================================================
//  SECTION 7: PPG (MAX30102) SETUP AND READING
//
//  The sensor shines a red LED into the skin and measures how much light
//  comes back. Every heart beat pushes more blood under the sensor, which
//  absorbs more light, so the reading dips a little with each beat.
//
//  The sensor stores samples in a 32-slot buffer (FIFO). Each loop reads
//  all new samples, so none are missed even when Bluetooth is busy.
// ============================================================================

const uint16_t PPG_SAMPLE_PERIOD_MS = 10;   // 400 samples/s averaged by 4 = 100 samples/s

void setupPpg() {
  uint8_t partId = readRegister(PPG_ADDR, 0xFF);   // MAX30102 part ID is 0x15
  if (partId != 0x15) {
    Serial.print("WARNING: MAX30102 not found at 0x57 (PART_ID = 0x");
    Serial.print(partId, HEX);
    Serial.println(")");
  }

  writeRegister(PPG_ADDR, 0x09, 0x40);  // MODE_CONFIG: reset the chip
  delay(100);

  writeRegister(PPG_ADDR, 0x08, 0x50);  // FIFO_CONFIG: average 4 samples, overwrite old data if full
  writeRegister(PPG_ADDR, 0x0A, 0x2F);  // SPO2_CONFIG: 4096 nA range, 400 samples/s, 18-bit
  writeRegister(PPG_ADDR, 0x0C, 0x1F);  // LED1 (red) current ≈ 6 mA
  writeRegister(PPG_ADDR, 0x09, 0x02);  // MODE_CONFIG: heart-rate mode (red LED only)

  writeRegister(PPG_ADDR, 0x04, 0x00);  // Clear FIFO write pointer
  writeRegister(PPG_ADDR, 0x05, 0x00);  // Clear overflow counter
  writeRegister(PPG_ADDR, 0x06, 0x00);  // Clear FIFO read pointer
}

// Forward declaration: the beat detector lives in Section 8.
void processPpgSample(uint32_t raw);

// Reads every new sample waiting in the FIFO and passes each one on.
void readPpgFifo() {
  uint8_t writePtr = readRegister(PPG_ADDR, 0x04);
  uint8_t readPtr  = readRegister(PPG_ADDR, 0x06);
  uint8_t available = (writePtr - readPtr) & 0x1F;   // FIFO has 32 slots

  while (available > 0) {
    // Read up to 8 samples (24 bytes) per I2C transfer.
    uint8_t count = min(available, (uint8_t)8);
    uint8_t buf[24];
    if (!readRegisters(PPG_ADDR, 0x07, buf, count * 3)) return;

    for (uint8_t i = 0; i < count; i++) {
      uint32_t sample = ((uint32_t)buf[3 * i] << 16) |
                        ((uint32_t)buf[3 * i + 1] << 8) |
                        buf[3 * i + 2];
      processPpgSample(sample & 0x3FFFF);   // Only the lower 18 bits are data
    }
    available -= count;
  }
}


// ============================================================================
//  SECTION 8: HEART RATE DETECTION
//
//  Step 1 - Skin check: a high average light level means skin is present.
//  Step 2 - Filtering:  subtract the slow average (removes the steady
//           level) and smooth it (removes fast noise). What's left is the
//           pulse wave, flipped so each beat is an upward peak.
//  Step 3 - Peak finding: a beat is a local peak that is at least half the
//           height of recent peaks.
//  Step 4 - Checking with the IMU: depending on the motion level from
//           Section 5, beats are accepted, checked strictly, or rejected.
//  Step 5 - BPM: the median of the last 8 beat gaps. The median ignores
//           occasional bad beats better than an average does.
// ============================================================================

uint32_t ppgClockMs   = 0;       // Time of the current sample (advances 10 ms per sample)
float    ppgDC        = 0;       // Slow average light level
float    ppgPulse     = 0;       // Filtered pulse wave
float    pulsePrev1   = 0;       // Pulse wave one sample ago
float    pulsePrev2   = 0;       // Pulse wave two samples ago
float    pulseEnvelope = 0;      // Height of recent peaks
bool     skinPresent  = false;

uint32_t lastBeatMs   = 0;       // Time of the last beat
uint32_t lastGoodBeatMs = 0;     // Time of the last ACCEPTED beat
uint16_t beatGaps[BEATS_TO_AVERAGE];   // Recent gaps between beats (ms)
uint8_t  beatGapCount = 0;
uint8_t  beatGapIndex = 0;
uint8_t  rejectsInRow = 0;

void clearBeatHistory() {
  beatGapCount = 0;
  beatGapIndex = 0;
  rejectsInRow = 0;
}

// Median of the stored beat gaps (copies and sorts them).
uint16_t medianBeatGap() {
  uint16_t sorted[BEATS_TO_AVERAGE];
  for (uint8_t i = 0; i < beatGapCount; i++) sorted[i] = beatGaps[i];

  for (uint8_t i = 1; i < beatGapCount; i++) {     // insertion sort
    uint16_t v = sorted[i];
    int8_t j = i - 1;
    while (j >= 0 && sorted[j] > v) { sorted[j + 1] = sorted[j]; j--; }
    sorted[j + 1] = v;
  }
  return sorted[beatGapCount / 2];
}

void acceptBeat(uint32_t beatMs, uint16_t gap) {
  beatGaps[beatGapIndex] = gap;
  beatGapIndex = (beatGapIndex + 1) % BEATS_TO_AVERAGE;
  if (beatGapCount < BEATS_TO_AVERAGE) beatGapCount++;

  lastBeatMs     = beatMs;
  lastGoodBeatMs = beatMs;
  rejectsInRow   = 0;

  if (beatGapCount >= 3) {
    currentBpm = (uint8_t)(60000UL / medianBeatGap());
  }
}

// Called for every beat-shaped peak. Decides whether it is a real beat.
void handlePeak(uint32_t beatMs) {
  uint32_t gap = beatMs - lastBeatMs;

  // Ignore peaks that come too soon after the last beat (noise / double peaks).
  if (gap < MIN_BEAT_GAP_MS) return;

  // Heavy motion: don't trust anything. Remember the time so the next gap
  // is measured from here, and keep showing the last good BPM.
  if (motionTooHigh(beatMs)) {
    lastBeatMs = beatMs;
    return;
  }

  // Long gap (first beat, or after skin contact came back): just start timing.
  if (gap > MAX_BEAT_GAP_MS) {
    lastBeatMs = beatMs;
    return;
  }

  // Not enough history to compare against yet: accept.
  if (beatGapCount < 3) {
    acceptBeat(beatMs, gap);
    return;
  }

  // Compare against the recent heart rate. Allow less variation while moving.
  uint16_t expected  = medianBeatGap();
  float    tolerance = (motionLevel == MOTION_MODERATE) ? 0.20 : 0.35;
  float    low       = expected * (1.0 - tolerance);
  float    high      = expected * (1.0 + tolerance);

  if (gap >= low && gap <= high) {
    acceptBeat(beatMs, gap);
    return;
  }

  // Rejected. If the gap was too LONG, we probably missed a beat, so restart
  // timing from here. If too SHORT, it was an extra bump; keep the old time.
  if (gap > high) lastBeatMs = beatMs;

  // Heart rate may have really changed (e.g. started exercising).
  // After several rejections in a row, start fresh.
  if (++rejectsInRow >= 5) clearBeatHistory();
}

// Raw PPG samples since the last CSV line, averaged when the line is sent.
// Averaging (instead of sending just the newest sample) smooths out noise
// that would otherwise alias when streaming slower than the sensor runs.
uint32_t ppgSum   = 0;
uint16_t ppgCount = 0;

void processPpgSample(uint32_t raw) {
  ppgClockMs += PPG_SAMPLE_PERIOD_MS;
  ppgSum += raw;
  ppgCount++;

  // ---- Step 1: Skin check ----
  if (raw < SKIN_THRESHOLD) {
    if (skinPresent) {
      skinPresent = false;
      clearBeatHistory();
      currentBpm = 0;
    }
    hrStatus = HR_NO_SKIN;
    return;
  }
  if (!skinPresent) {
    // Skin just appeared: start the filters at the current level so they
    // don't take seconds to catch up.
    skinPresent   = true;
    ppgDC         = raw;
    ppgPulse      = pulsePrev1 = pulsePrev2 = 0;
    pulseEnvelope = 0;
    lastBeatMs    = ppgClockMs;
    lastGoodBeatMs = ppgClockMs;
  }

  // ---- Step 2: Filtering ----
  ppgDC += 0.02 * (raw - ppgDC);          // slow average (≈ steady light level)
  float ac = ppgDC - raw;                 // flipped so a beat is a peak
  ppgPulse += 0.3 * (ac - ppgPulse);      // smooth out fast noise

  // Track the height of recent peaks (slowly shrinks if peaks get smaller).
  pulseEnvelope *= 0.995;
  if (fabsf(ppgPulse) > pulseEnvelope) pulseEnvelope = fabsf(ppgPulse);

  // ---- Step 3: Peak finding ----
  // The previous sample is a peak if it is higher than both neighbours and
  // tall enough compared to recent peaks.
  bool isPeak = pulsePrev1 > pulsePrev2 &&
                pulsePrev1 >= ppgPulse &&
                pulsePrev1 > 0.5 * pulseEnvelope;
  if (isPeak) {
    handlePeak(ppgClockMs - PPG_SAMPLE_PERIOD_MS);   // Steps 4 + 5
  }

  pulsePrev2 = pulsePrev1;
  pulsePrev1 = ppgPulse;

  // ---- Update status ----
  if (motionTooHigh(ppgClockMs)) {
    hrStatus = HR_MOTION;
  } else if (ppgClockMs - lastGoodBeatMs > HR_TIMEOUT_MS) {
    currentBpm = 0;                       // No good beats for too long
    hrStatus = HR_SEARCHING;
  } else {
    hrStatus = (currentBpm > 0) ? HR_OK : HR_SEARCHING;
  }
}


// ============================================================================
//  SECTION 9: BLUETOOTH (Nordic UART Service)
//
//  NUS is a standard "serial port over Bluetooth". The band sends text on
//  its TX characteristic and collect.py subscribes to it. Lines longer than
//  one Bluetooth packet are split automatically and collect.py joins them
//  back together at each newline.
// ============================================================================

BLEUart bleuart;

void onConnect(uint16_t connHandle) {
  // Ask for bigger packets so each CSV line fits in one or two of them.
  Bluefruit.Connection(connHandle)->requestMtuExchange(247);
  Serial.println("# collect.py connected");
}

void onDisconnect(uint16_t connHandle, uint8_t reason) {
  Serial.println("# collect.py disconnected");
}

void setupBluetooth() {
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);   // allow large packets (must come before begin)
  Bluefruit.begin();
  Bluefruit.setTxPower(4);
  Bluefruit.setName(DEVICE_NAME);
  Bluefruit.Periph.setConnectCallback(onConnect);
  Bluefruit.Periph.setDisconnectCallback(onDisconnect);

  bleuart.begin();

  // ---- Advertising (how the computer finds the band) ----
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);   // advertise again after a disconnect
  Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);                    // 0 = advertise forever
}


// ============================================================================
//  SECTION 10: BUILD AND SEND ONE CSV LINE
// ============================================================================

void sendCsvLine(uint32_t nowMs) {
  // Average the raw PPG samples collected since the last line.
  uint32_t ppgAvg = (ppgCount > 0) ? ppgSum / ppgCount : 0;
  ppgSum = 0;
  ppgCount = 0;

  // Build the line as text: ms,ppg,pulse,ax_mg,ay_mg,az_mg,gx_dps,gy_dps,gz_dps,motion,bpm,status,steps
  String line;
  line.reserve(120);
  line += nowMs;                             line += ',';
  line += ppgAvg;                            line += ',';
  line += String(ppgPulse, 1);               line += ',';
  line += String(latestImu.ax * 1000, 1);    line += ',';
  line += String(latestImu.ay * 1000, 1);    line += ',';
  line += String(latestImu.az * 1000, 1);    line += ',';
  line += String(latestImu.gx, 2);           line += ',';
  line += String(latestImu.gy, 2);           line += ',';
  line += String(latestImu.gz, 2);           line += ',';
  line += (int)motionLevel;                  line += ',';
  line += currentBpm;                        line += ',';
  line += (int)hrStatus;                     line += ',';
  line += stepsToday;
  line += '\n';

  if (Bluefruit.connected() && bleuart.notifyEnabled()) {
    bleuart.print(line);
  }
  if (ECHO_TO_SERIAL) {
    Serial.print(line);
  }
}


// ============================================================================
//  SECTION 11: SETUP  (runs once at power-on)
// ============================================================================

void setup() {
  Serial.begin(115200);       // no waiting for Serial: the band must run on battery too
  Wire.begin();
  Wire.setClock(400000);      // fast I2C (400 kHz) so sensor reads stay quick

  setupImu();
  setupPpg();
  setupBluetooth();

  Serial.println("# FitBand ready, waiting for collect.py");
}


// ============================================================================
//  SECTION 12: MAIN LOOP  (runs over and over)
//
//  Every 10 ms               : read the IMU -> motion level and step counter
//  Every loop                : read all new PPG samples -> beat detection
//  Every STREAM_PERIOD_MS    : send one CSV line
// ============================================================================

uint32_t lastImuMs    = 0;
uint32_t lastStreamMs = 0;

void loop() {
  uint32_t now = millis();

  // ---- IMU: motion + steps, 100 times per second ----
  if (now - lastImuMs >= 10) {
    lastImuMs = now;
    if (readImu(latestImu)) {
      updateMotion(latestImu, now);
      updateSteps(now);
    }
  }

  // ---- PPG: heart rate ----
  // Keep the PPG clock in step with the real clock (it can drift slightly).
  if ((int32_t)(now - ppgClockMs) > 500 || (int32_t)(ppgClockMs - now) > 500) {
    ppgClockMs = now;
  }
  readPpgFifo();

  // ---- Stream one CSV line ----
  if (now - lastStreamMs >= STREAM_PERIOD_MS) {
    lastStreamMs = now;
    sendCsvLine(now);
  }
}
