# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
from app.base.resources import ResourceManager
from threading import Thread


@pytest.fixture
def resource_manager():
    """Fixture to create a ResourceManager instance with 1GB GPU and CPU memory"""
    total_gpu = 1024 * 1024 * 1024  # 1GB in bytes
    total_cpu = 1024 * 1024 * 1024
    return ResourceManager(total_gpu, total_cpu), total_gpu, total_cpu


def test_initialization(resource_manager):
    """Test proper initialization of ResourceManager"""
    manager, total_gpu, total_cpu = resource_manager
    assert manager.get_available_gpu_memory() == total_gpu
    assert manager.get_available_cpu_memory() == total_cpu


def test_successful_allocation(resource_manager):
    """Test successful memory allocation"""
    manager, total_gpu, total_cpu = resource_manager
    allocation_size = 512 * 1024 * 1024  # 512MB

    # Check if allocation is possible
    assert manager.can_allocate(allocation_size, allocation_size)

    # Perform allocation
    success = manager.allocate(allocation_size, allocation_size)
    assert success

    # Verify remaining memory
    assert manager.get_available_gpu_memory() == total_gpu - allocation_size
    assert manager.get_available_cpu_memory() == total_cpu - allocation_size


def test_failed_allocation(resource_manager):
    """Test allocation failure when requesting more than available memory"""
    manager, total_gpu, _ = resource_manager
    excessive_size = total_gpu + 1

    # Check if allocation is possible
    assert not manager.can_allocate(excessive_size, 0)

    # Attempt allocation
    success = manager.allocate(excessive_size, 0)
    assert not success

    # Verify memory hasn't changed
    assert manager.get_available_gpu_memory() == total_gpu


def test_memory_release(resource_manager):
    """Test memory release functionality"""
    manager, total_gpu, total_cpu = resource_manager
    allocation_size = 256 * 1024 * 1024  # 256MB

    # Allocate memory
    manager.allocate(allocation_size, allocation_size)

    # Release memory
    manager.release(allocation_size, allocation_size)

    # Verify all memory is available again
    assert manager.get_available_gpu_memory() == total_gpu
    assert manager.get_available_cpu_memory() == total_cpu


def test_multiple_allocations(resource_manager):
    """Test multiple sequential allocations"""
    manager, total_gpu, total_cpu = resource_manager
    allocation_size = 256 * 1024 * 1024  # 256MB

    # Perform 3 allocations
    for _ in range(3):
        success = manager.allocate(allocation_size, allocation_size)
        assert success

    # Verify remaining memory
    expected_remaining = total_gpu - (3 * allocation_size)
    assert manager.get_available_gpu_memory() == expected_remaining
    assert manager.get_available_cpu_memory() == expected_remaining


def test_thread_safety(resource_manager):
    """Test thread-safe operations"""
    manager, total_gpu, total_cpu = resource_manager
    allocation_size = 100 * 1024 * 1024  # 100MB
    num_threads = 10

    def allocate_and_release():
        manager.allocate(allocation_size, allocation_size)
        manager.release(allocation_size, allocation_size)

    threads = [Thread(target=allocate_and_release) for _ in range(num_threads)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify final memory state
    assert manager.get_available_gpu_memory() == total_gpu
    assert manager.get_available_cpu_memory() == total_cpu
