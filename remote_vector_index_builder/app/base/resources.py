# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import threading


class ResourceManager:
    """
    A thread-safe resource manager that tracks and manages GPU and CPU memory allocations.

    This class provides mechanisms to safely allocate and release memory resources
    in a multi-threaded environment, ensuring that memory usage doesn't exceed
    the specified limits.

    Attributes:
        _total_gpu_memory (float): Total available GPU memory in bytes
        _total_cpu_memory (float): Total available CPU memory in bytes
        _available_gpu_memory (float): Currently available GPU memory in bytes
        _available_cpu_memory (float): Currently available CPU memory in bytes
        _lock (threading.Lock): Thread lock for synchronization
    """

    def __init__(self, total_gpu_memory: float, total_cpu_memory: float):
        """
        Initialize the ResourceManager with specified GPU and CPU memory limits.

        Args:
            total_gpu_memory (float): Total GPU memory available for allocation, in bytes
            total_cpu_memory (float): Total CPU memory available for allocation, in bytes
        """
        self._total_gpu_memory = total_gpu_memory
        self._total_cpu_memory = total_cpu_memory
        self._available_gpu_memory = total_gpu_memory
        self._available_cpu_memory = total_cpu_memory
        self._lock = threading.Lock()

    def allocate(self, gpu_memory: float, cpu_memory: float) -> bool:
        """
        Attempt to allocate the specified amount of GPU and CPU memory.

        Args:
            gpu_memory (float): Amount of GPU memory to allocate, in bytes
            cpu_memory (float): Amount of CPU memory to allocate, in bytes

        Returns:
            bool: True if allocation was successful, False if insufficient resources
        """
        with self._lock:
            if not (self._available_gpu_memory >= gpu_memory and self._available_cpu_memory >= cpu_memory):
                return False
            self._available_gpu_memory -= gpu_memory
            self._available_cpu_memory -= cpu_memory
            return True

    def release(self, gpu_memory: float, cpu_memory: float) -> None:
        """
        Release previously allocated GPU and CPU memory back to the pool.

        Args:
            gpu_memory (float): Amount of GPU memory to release, in bytes
            cpu_memory (float): Amount of CPU memory to release, in bytes
        """
        with self._lock:
            self._available_gpu_memory += gpu_memory
            self._available_cpu_memory += cpu_memory

    def get_available_gpu_memory(self) -> float:
        """
        Get the current amount of available GPU memory.

        Returns:
            float: Amount of available GPU memory in bytes
        """
        with self._lock:
            return self._available_gpu_memory

    def get_available_cpu_memory(self) -> float:
        """
        Get the current amount of available GPU memory.

        Returns:
            float: Amount of available GPU memory in bytes
        """
        with self._lock:
            return self._available_cpu_memory
