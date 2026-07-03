"""
System Information

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import platform
import psutil
import cv2
import torch
import cpuinfo


class SystemInfo:

    @staticmethod
    def collect():

        cpu = cpuinfo.get_cpu_info()

        info = {

            "cpu": cpu["brand_raw"],

            "architecture": platform.machine(),

            "physical_cores": psutil.cpu_count(logical=False),

            "logical_cores": psutil.cpu_count(logical=True),

            "ram_gb": round(
                psutil.virtual_memory().total / (1024**3), 2
            ),

            "python": platform.python_version(),

            "opencv": cv2.__version__,

            "pytorch": torch.__version__,

            "cuda": torch.cuda.is_available(),

            "os": platform.system(),

            "release": platform.release(),

            "platform": platform.platform(),

            "processor": platform.processor()

        }

        return info

    @staticmethod
    def print():

        info = SystemInfo.collect()

        print("=" * 60)
        print("System Information")
        print("=" * 60)

        for key, value in info.items():
            print(f"{key:20}: {value}")

        print("=" * 60)