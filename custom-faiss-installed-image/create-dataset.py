import numpy as np

if __name__ == "__main__":
    d = 768
    np.random.seed(1234)
    # Create a new memory-mapped array
    shape = (7_000_000, 768)  # Example shape
    print(f"Creating dataset of {shape}")
    filename = 'mmap-7m.npy'

    mmap_array = np.lib.format.open_memmap(filename, mode='w+', dtype='float32', shape=shape)
    mmap_array[:] = np.random.rand(*shape)
    del mmap_array
    print("file is written")
