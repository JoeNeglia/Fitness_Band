"""
STEP 1 — RECORD A TRAINING SESSION

Plug in the band (running Arduino/data_logger), wear a Bluetooth chest strap, then:

    .venv/bin/python record.py joe              # full ~30-min protocol
    .venv/bin/python record.py joe --quick      # 3-minute test run

Walks you through the paper's activities, and saves to data/raw/<date>_<name>/:
    band.csv       raw band data
    reference.csv  chest strap heart rate (the "true" heart rate to learn from)
    meta.json      start/end time of each activity
"""

import argparse
import asyncio
import csv
import json
import threading
import time
from datetime import datetime

import serial
from bleak import BleakClient, BleakScanner
from serial.tools import list_ports

from common import ACTIVITY_CLASS, PROTOCOL, RAW_DIR

INSTRUCTIONS = {
    "rest": "Sit still, band hand resting on the table.",
    "read": "Sit and turn a book page with the band hand about every 5 seconds.",
    "write": "Sit and write or draw on paper with the band hand.",
    "walk_slow": "Walk slowly, arms swinging naturally.",
    "walk_fast": "Walk briskly, arms swinging naturally.",
}


# =============================================================================
#  BAND: reads CSV lines from USB and saves them with the laptop's clock time
# =============================================================================
def record_band(path, stop):
    port = next((p.device for p in list_ports.comports() if "usbmodem" in p.device), None)
    if port is None:
        print("\nBand not found! Plug it in and upload data_logger.ino.")
        return
    with serial.Serial(port, 115200, timeout=1) as ser, open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["host_time", "t_ms", "red", "ir", "ax", "ay", "az", "gx", "gy", "gz"])
        start = None
        while not stop.is_set():
            parts = ser.readline().decode(errors="ignore").strip().split(",")
            if len(parts) != 9 or not parts[0].isdigit():
                continue                                # skip messages / partial lines
            t_ms = int(parts[0])
            if start is None:
                start = time.time() - t_ms / 1000       # laptop time when band time was 0
            out.writerow([f"{start + t_ms / 1000:.3f}"] + parts)


# =============================================================================
#  CHEST STRAP: standard Bluetooth heart rate service, 1 reading per second
# =============================================================================
HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
latest_strap_hr = None


def record_strap(path, stop):
    async def run():
        global latest_strap_hr
        strap = await BleakScanner.find_device_by_filter(
            lambda d, adv: HR_SERVICE in [u.lower() for u in adv.service_uuids], timeout=20)
        if strap is None:
            print("\nNo chest strap found! Is it on and moistened?")
            return
        with open(path, "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(["time", "hr"])

            def on_reading(_, data):
                global latest_strap_hr
                # Byte 0 = flags. Bit 0 says whether the heart rate is 1 or 2 bytes.
                hr = int.from_bytes(data[1:3], "little") if data[0] & 1 else data[1]
                latest_strap_hr = hr
                out.writerow([f"{time.time():.3f}", hr])

            async with BleakClient(strap) as client:
                await client.start_notify(HR_MEASUREMENT, on_reading)
                while not stop.is_set():
                    await asyncio.sleep(0.5)

    asyncio.run(run())


# =============================================================================
#  PROTOCOL
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="who is wearing the band")
    ap.add_argument("--quick", action="store_true", help="10x shorter activities, for testing")
    args = ap.parse_args()

    folder = RAW_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}"
    folder.mkdir(parents=True)
    stop = threading.Event()
    threading.Thread(target=record_band, args=(folder / "band.csv", stop), daemon=True).start()
    threading.Thread(target=record_strap, args=(folder / "reference.csv", stop), daemon=True).start()
    print(f"Saving to {folder}\nConnecting to band and chest strap...")

    segments = []
    try:
        for i, (activity, minutes) in enumerate(PROTOCOL, 1):
            seconds = minutes * 60 / (10 if args.quick else 1)
            print(f"\n[{i}/{len(PROTOCOL)}] {activity.upper()} for {seconds / 60:.1f} min")
            print(f"    {INSTRUCTIONS[activity]}")
            input("    Press Enter to start...")
            start = time.time()
            while (left := start + seconds - time.time()) > 0:
                print(f"\r    {int(left) // 60:02d}:{int(left) % 60:02d} left   strap: {latest_strap_hr} BPM  ",
                      end="", flush=True)
                time.sleep(0.5)
            print("\a")
            segments.append({"activity": activity, "class": ACTIVITY_CLASS[activity],
                             "start": start, "end": time.time()})
        print("\nAll done!")
    except KeyboardInterrupt:
        print("\nStopped early; finished activities are kept.")
    finally:
        stop.set()
        (folder / "meta.json").write_text(json.dumps({"segments": segments}, indent=2))
        time.sleep(1)


if __name__ == "__main__":
    main()
