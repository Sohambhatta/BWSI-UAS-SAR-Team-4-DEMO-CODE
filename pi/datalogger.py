import time
import json
import struct
import os
import numpy as np

class Datalogger:
    def __init__(self):
        self.t = time.time()
        self.directory = os.path.join("DATAS", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self.t)))
        os.makedirs(self.directory, exist_ok=True)

    def write_metadata(self, scan_start, scan_end, scan_res):
        """Writes metadata about the scan to a file and to the first USB drive at /mnt/sardemo/usb if available."""
        metadata = {
            "timestamp": self.t,
            "scan_start": scan_start,
            "scan_end": scan_end,
            "scan_res": scan_res,
        }
        # Write to local directory
        os.makedirs(self.directory, exist_ok=True)
        with open(os.path.join(self.directory, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        # Write to USB if available
        usb_base = "/media/sardemo/USB"
        if os.path.exists(usb_base):
            usb_dir = os.path.join(usb_base, os.path.basename(self.directory))
            os.makedirs(usb_dir, exist_ok=True)
            with open(os.path.join(usb_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

    def write_scan(self, full_data):
        """Writes the scan data to a binary file and to the first USB drive at /mnt/sardemo/usb if available."""
        # Write to local directory
        os.makedirs(self.directory, exist_ok=True)
        with open(os.path.join(self.directory, "returns.bin"), "wb") as f:
            for d in full_data:
                millis_time = np.int32(d[0])
                data = d[1].astype(np.int32)
                arr = np.concatenate((np.array([len(data)], dtype=np.int32), np.array([millis_time], dtype=np.int32), data))
                arr.tofile(f)
        # Write to USB if available
        usb_base = "/media/sardemo/USB"
        if os.path.exists(usb_base):
            usb_dir = os.path.join(usb_base, os.path.basename(self.directory))
            os.makedirs(usb_dir, exist_ok=True)
            with open(os.path.join(usb_dir, "returns.bin"), "wb") as f:
                for d in full_data:
                    millis_time = np.int32(d[0])
                    data = d[1].astype(np.int32)
                    arr = np.concatenate((np.array([len(data)], dtype=np.int32), np.array([millis_time], dtype=np.int32), data))
                    arr.tofile(f)
