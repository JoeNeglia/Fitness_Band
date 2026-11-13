"""
collect.py — record one activity block from the band and the chest strap.

    python collect/collect.py <activity> <seconds>
    python collect/collect.py test 60
    python collect/collect.py walking 300

Connects over Bluetooth to:
  - the FitBand (Nordic UART Service): one CSV line about 10 times per second
  - the Garmin HRM 200 chest strap (standard Heart Rate Service): heart rate
    about twice per second, plus beat-to-beat (RR) intervals when available

Every row is stamped with this computer's clock (pc_time, seconds since
1970). Both devices are stamped by the same clock, which is how the band
and strap data are lined up later.

Writes two files into data/raw/ (move them into data/raw/sessionN/ after
the session):
    band_<activity>_<YYYYMMDD_HHMMSS>.csv
    hrm_<activity>_<YYYYMMDD_HHMMSS>.csv
"""

import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

# =============================================================================
#  SETTINGS
# =============================================================================
BAND_NAME = "FitBand"                        # name set in fit_band_ble.ino
SCAN_TIMEOUT_S = 20                          # how long to look for each device
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Nordic UART Service: the band sends text on the TX characteristic
UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# Standard Bluetooth Heart Rate Service (any chest strap)
HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"

# Columns sent by the band (13), plus pc_time added here
BAND_FIELDS = ["ms", "ppg", "pulse", "ax_mg", "ay_mg", "az_mg",
               "gx_dps", "gy_dps", "gz_dps", "motion", "bpm", "status", "steps"]
BAND_HEADER = ["pc_time"] + BAND_FIELDS
HRM_HEADER = ["pc_time", "hr", "rr_ms"]


# =============================================================================
#  BAND: join Bluetooth packets into lines, write one row per line
# =============================================================================
class BandRecorder:
    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(BAND_HEADER)
        self.buffer = ""          # text received but not yet ending in a newline
        self.rows = 0
        self.last_bpm = None

    def on_packet(self, _sender, data):
        """Called by bleak for every Bluetooth packet. A line may span several packets."""
        pc_time = time.time()
        self.buffer += data.decode(errors="ignore")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.write_line(line.strip(), pc_time)

    def write_line(self, line, pc_time):
        values = line.split(",")
        if len(values) != len(BAND_FIELDS):
            return                # skip partial or garbled lines
        self.writer.writerow([f"{pc_time:.3f}"] + values)
        self.rows += 1
        self.last_bpm = values[BAND_FIELDS.index("bpm")]

    def close(self):
        self.file.close()


# =============================================================================
#  CHEST STRAP: decode the standard Heart Rate Measurement packet
# =============================================================================
def parse_heart_rate(data):
    """
    Returns (heart_rate_bpm, [rr_ms, ...]).
    Byte 0 is a set of flags:
      bit 0: heart rate is 16-bit (else 8-bit)
      bit 3: energy expended field present (2 bytes, skipped)
      bit 4: RR intervals present (2 bytes each, in units of 1/1024 s)
    """
    flags = data[0]
    if flags & 0x01:
        hr = int.from_bytes(data[1:3], "little")
        i = 3
    else:
        hr = data[1]
        i = 2
    if flags & 0x08:
        i += 2
    rr = []
    if flags & 0x10:
        while i + 1 < len(data):
            rr.append(int.from_bytes(data[i:i + 2], "little") * 1000 / 1024)
            i += 2
    return hr, rr


class HrmRecorder:
    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(HRM_HEADER)
        self.rows = 0
        self.last_hr = None

    def on_packet(self, _sender, data):
        pc_time = time.time()
        hr, rr = parse_heart_rate(data)
        # Several RR values in one packet are joined with ';' (oldest first)
        rr_text = ";".join(f"{r:.0f}" for r in rr)
        self.writer.writerow([f"{pc_time:.3f}", hr, rr_text])
        self.rows += 1
        self.last_hr = hr

    def close(self):
        self.file.close()


# =============================================================================
#  BLUETOOTH: find both devices, subscribe, record for N seconds
# =============================================================================
def is_band(device, adv):
    return device.name == BAND_NAME or UART_SERVICE in [u.lower() for u in adv.service_uuids]


def is_strap(device, adv):
    return HR_SERVICE in [u.lower() for u in adv.service_uuids]


async def find(name, match):
    print(f"Looking for {name}...")
    device = await BleakScanner.find_device_by_filter(match, timeout=SCAN_TIMEOUT_S)
    if device is None:
        sys.exit(f"Could not find the {name}. Is it on and nearby?")
    print(f"  found {device.name or device.address}")
    return device


async def subscribe_to_strap(strap_client, on_packet, attempts=4):
    """
    The HRM 200 only sends heart rate over a paired (encrypted) connection.
    The first time, macOS fails with "Encryption is insufficient" while it
    shows a pairing request. After you accept it, trying again works.
    """
    for attempt in range(1, attempts + 1):
        try:
            if not strap_client.is_connected:      # a failed pairing makes the strap hang up
                await strap_client.connect()
            await strap_client.start_notify(HR_MEASUREMENT, on_packet)
            return
        except BleakError as error:
            if attempt == attempts:
                sys.exit("\nThe strap refused to pair with this Mac. See 'Strap won't pair' in "
                         "docs/Data_Collection_Guide.pdf.")
            print(f"  Strap not paired yet ({error}).")
            print("  If a Bluetooth pairing request pops up on your Mac, click Connect/Pair."
                  f" Retrying in 5 s ({attempt}/{attempts - 1})...")
            await asyncio.sleep(5)


async def record(activity, seconds):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    band = BandRecorder(OUTPUT_DIR / f"band_{activity}_{stamp}.csv")
    hrm = HrmRecorder(OUTPUT_DIR / f"hrm_{activity}_{stamp}.csv")

    try:
        band_device = await find("band", is_band)
        strap_device = await find("chest strap", is_strap)

        async with BleakClient(band_device) as band_client, BleakClient(strap_device) as strap_client:
            # Strap first, so band rows only start once both devices are ready
            await subscribe_to_strap(strap_client, hrm.on_packet)
            await band_client.start_notify(UART_TX, band.on_packet)
            print(f"\nRecording '{activity}' for {seconds} s. Press Ctrl-C to stop early.\n")

            start = time.time()
            while time.time() - start < seconds:
                await asyncio.sleep(5)
                elapsed = int(time.time() - start)
                print(f"  {elapsed:>4}s  band rows {band.rows:>5} (bpm {band.last_bpm})   "
                      f"strap rows {hrm.rows:>4} (hr {hrm.last_hr})")
                if not band_client.is_connected or not strap_client.is_connected:
                    print("  A device disconnected, stopping.")
                    break
    finally:
        # Runs even after Ctrl-C, so everything recorded so far is saved.
        band.close()
        hrm.close()
        print(f"\nSaved {band.rows} band rows and {hrm.rows} strap rows to {OUTPUT_DIR}")


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3 or not sys.argv[2].isdigit():
        sys.exit("Usage: python collect/collect.py <activity> <seconds>   e.g. walking 300")
    try:
        asyncio.run(record(sys.argv[1], int(sys.argv[2])))
    except KeyboardInterrupt:
        print("\nStopped early. Rows recorded so far are saved.")
