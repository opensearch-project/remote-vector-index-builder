def test_imports():
    try:
        import numpy
        import tqdm
        import yaml
        import h5py
        from benchmarking.workload.workload import runWorkload

        print("All imports successful!")
        return 0
    except ImportError as e:
        print(f"Import failed: {e}")
        exit(1)


if __name__ == "__main__":
    test_imports()
