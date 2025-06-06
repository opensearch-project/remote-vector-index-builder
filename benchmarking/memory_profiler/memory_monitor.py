from py3nvml import nvidia_smi
import psutil
import time
import threading
import pandas as pd
from typing import List
import logging


class MemoryMonitor:
    def __init__(self, id: str, interval: float = 0.1, monitor_gpu=True):
        self.interval = interval
        self.cpu_memory_logs = []
        self.start_time = None
        self.ram_used_mb: List[float] = []
        self.timestamps: List[float] = []
        self.is_monitoring = False
        self._monitor_thread = None
        self.process = psutil.Process()
        self.id = id

        # Initialize GPU monitoring
        if monitor_gpu:
            self.monitor_gpu = True
            self.gpu_id = 0
            self.gpu_memory_logs = []
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu_id)
        else:
            self.monitor_gpu = False

    def _get_cpu_memory_info(self):
        """Get current CPU memory usage in MB"""
        cpu_info = psutil.virtual_memory()
        return {
            "used": cpu_info.used / 1024 / 1024,  # Convert to MB
            "free": cpu_info.available / 1024 / 1024,
            "total": cpu_info.total / 1024 / 1024,
        }

    def _get_gpu_memory_info(self):
        """Get current GPU memory usage in MB"""
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "used": info.used / 1024 / 1024,  # Convert to MB
            "free": info.free / 1024 / 1024,
            "total": info.total / 1024 / 1024,
        }

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            self.timestamps.append(time.time())

            if self.monitor_gpu:
                gpu_memory_info = self._get_gpu_memory_info()

                self.gpu_memory_logs.append(
                    {
                        "gpu_used_memory": gpu_memory_info["used"],
                        "gpu_free_memory": gpu_memory_info["free"],
                        "gpu_total_memory": gpu_memory_info["total"],
                        "id": self.id,
                    }
                )

            cpu_memory_info = self._get_cpu_memory_info()
            self.cpu_memory_logs.append(
                {
                    "cpu_used_memory": cpu_memory_info["used"],
                    "cpu_free_memory": cpu_memory_info["free"],
                    "cpu_total_memory": cpu_memory_info["total"],
                    "id": self.id,
                }
            )
            self.ram_used_mb.append(self._get_process_memory_mb())
            time.sleep(self.interval)  # sampling rate

    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in KB"""
        return self.process.memory_info().rss / (1024**2)  # Convert bytes to MB

    def start_monitoring(self):
        """Start GPU memory monitoring"""
        self.start_time = time.time()
        self.is_monitoring = True
        self.gpu_memory_logs = []
        self.cpu_memory_logs = []
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop GPU memory monitoring"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def log_gpu_metrics(self):
        """Log GPU memory usage metrics"""
        if self.monitor_gpu:
            df = pd.DataFrame(self.gpu_memory_logs)
            max_memory = df["gpu_used_memory"].max()
            start_memory = df["gpu_used_memory"].iloc[0]
            end_memory = df["gpu_used_memory"].iloc[-1]
            logging.info(f"Start GPU Memory: ,{start_memory}")
            logging.info(f"End GPU Memory: ,{end_memory}")
            logging.info(f"Max GPU Memory: ,{max_memory}")
            logging.info(f"Net GPU Memory used:, {max_memory-start_memory}")
            return max_memory, start_memory, end_memory
        return 0, 0, 0

    def log_cpu_metrics(self):
        """Log CPU memory usage metrics"""
        df = pd.DataFrame(self.cpu_memory_logs)
        max_memory = df["cpu_used_memory"].max()
        start_memory = df["cpu_used_memory"].iloc[0]
        end_memory = df["cpu_used_memory"].iloc[-1]
        logging.info(f"Start CPU Memory: ,{start_memory}")
        logging.info(f"End CPU Memory: ,{end_memory}")
        logging.info(f"Max CPU Memory: ,{max_memory}")
        logging.info(f"Net CPU Memory used:, {max_memory-start_memory}")
        return max_memory, start_memory, end_memory

    def __del__(self):
        """Cleanup NVML on object destruction"""
        try:
            if self.monitor_gpu:
                nvidia_smi.nvmlShutdown()
        except Exception:
            pass
