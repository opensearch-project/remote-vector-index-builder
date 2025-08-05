import numpy as np

from remote_vector_index_builder.core.fp32_to_fp16_converting_bytes_io import (
    FP32ToFP16ConvertingBytesIO,
)

dimension = 64
n_vectors = 100
total_vectors = dimension * n_vectors


def _test_fp32_to_fp16_conversion(fp32: np.ndarray, fp16: np.ndarray):
    assert fp32.dtype == np.float32, "Input array 'fp32' must be of dtype float32"
    assert fp16.dtype == np.float16, "Input array 'fp16' must be of dtype float16"
    assert fp32.shape == fp16.shape, "Input arrays must have the same shape"

    expected_fp16 = fp32.astype(np.float16)
    np.testing.assert_array_equal(fp16, expected_fp16)


def test1_perfect_match():
    bytes_io = FP32ToFP16ConvertingBytesIO(total_vectors)
    fp32_values = np.random.uniform(-100, 100, size=total_vectors).astype(np.float32)
    bytes_to_write = fp32_values.tobytes()

    # Write first 10 vectors
    bytes_io.seek(0)
    ret = bytes_io.write(bytes_to_write[0:40])
    assert ret == 40

    _test_fp32_to_fp16_conversion(fp32_values[0:10], bytes_io._fp16_np[0:10])

    # write 10 vectors from third.
    bytes_io = FP32ToFP16ConvertingBytesIO(total_vectors)
    bytes_io.seek(0)
    ret = bytes_io.write(bytes_to_write[4 * 3 : 4 * 3 + 40])
    assert ret == 40

    _test_fp32_to_fp16_conversion(fp32_values[3:13], bytes_io._fp16_np[0:10])


def test2_incomplete_bytes_leading():
    offsets = [1, 2, 3]
    for o in offsets:
        bytes_io = FP32ToFP16ConvertingBytesIO(total_vectors)
        fp32_values = np.random.uniform(-100, 100, size=total_vectors).astype(
            np.float32
        )
        bytes_to_write = fp32_values.tobytes()
        org_bytes_to_write = bytes_to_write

        # Write 'incomplete bytes of 4th vector' + '10 vectors' after
        # e.g. 'incomplete bytes of 4th vector', fp16 <- fp32[5:15]
        write_start_offset = 20 - o
        bytes_io.seek(write_start_offset)
        bytes_to_write = bytes_to_write[
            write_start_offset : write_start_offset + o + 4 * 10
        ]
        ret = bytes_io.write(bytes_to_write)
        assert ret == o + 4 * 10

        # Check converted fp16 vectors
        _test_fp32_to_fp16_conversion(fp32_values[5:15], bytes_io._fp16_np[5:15])

        # Check if it has incomplete bytes correctly
        vector_idx = 4
        assert vector_idx in bytes_io._incomplete_vector_value
        assert bytes_io._incomplete_vector_value[vector_idx]["count"] == o
        j = 0  # Index of bytes_to_write
        i = 4 - o
        while i < 4:
            assert (
                bytes_io._incomplete_vector_value[vector_idx]["bytes"][i]
                == bytes_to_write[j]
            )
            j += 1
            i += 1

        # Write remaining. e.g. 'incomplete bytes of 4th vector' + fp16 <- fp32[0:4]
        bytes_to_write = org_bytes_to_write[0:write_start_offset]
        bytes_io.seek(0)
        ret = bytes_io.write(bytes_to_write)
        assert ret == write_start_offset

        # Expect it should complete 4th vector.
        _test_fp32_to_fp16_conversion(fp32_values[0:5], bytes_io._fp16_np[0:5])

        assert vector_idx not in bytes_io._incomplete_vector_value


def test3_incomplete_bytes_trailing():
    offsets = [1, 2, 3]
    for o in offsets:
        bytes_io = FP32ToFP16ConvertingBytesIO(total_vectors)
        fp32_values = np.random.uniform(-100, 100, size=total_vectors).astype(
            np.float32
        )
        bytes_to_write = fp32_values.tobytes()
        org_bytes_to_write = bytes_to_write

        # Write '10 vectors' + 'incomplete bytes of 5th vector' after 4 vectors.
        # e.g. fp16 <- fp32[4:14], and one incomplete bytes
        write_start_offset = 16
        bytes_io.seek(write_start_offset)
        bytes_to_write = bytes_to_write[
            write_start_offset : write_start_offset + 4 * 10 + o
        ]
        ret = bytes_io.write(bytes_to_write)
        assert ret == o + 4 * 10

        # Check converted fp16 vectors
        _test_fp32_to_fp16_conversion(fp32_values[4:14], bytes_io._fp16_np[4:14])

        # Check if it has incomplete bytes correctly
        vector_idx = 14
        assert vector_idx in bytes_io._incomplete_vector_value
        assert bytes_io._incomplete_vector_value[vector_idx]["count"] == o
        j = 4 * 10  # Index of bytes_to_write
        i = 0
        while i < o:
            assert (
                bytes_io._incomplete_vector_value[vector_idx]["bytes"][i]
                == bytes_to_write[j]
            )
            j += 1
            i += 1

        # Write 'incomplete bytes in 14th vector' + 10 vectors.
        # e.g. 'incomplete bytes in 14th vector', fp16 <- fp32[15:25]
        s = write_start_offset + 4 * 10 + o
        bytes_to_write = org_bytes_to_write[s : s + 4 - o + 4 * 10]
        ret = bytes_io.write(bytes_to_write)
        assert ret == 4 * 10 + 4 - o
        _test_fp32_to_fp16_conversion(fp32_values[14:25], bytes_io._fp16_np[14:25])

        assert vector_idx not in bytes_io._incomplete_vector_value


def test4_incomplete_bytes_leading_trailing():
    offsets = [1, 2, 3]
    for o in offsets:
        for o2 in offsets:
            bytes_io = FP32ToFP16ConvertingBytesIO(total_vectors)
            fp32_values = np.random.uniform(-100, 100, size=total_vectors).astype(
                np.float32
            )
            bytes_to_write = fp32_values.tobytes()

            # Convert fp32[5:15] + incomplete bytes in 4th + incomplete bytes in 15th
            write_start_offset = 20 - o
            bytes_io.seek(write_start_offset)
            bytes_to_write = bytes_to_write[
                write_start_offset : write_start_offset + o + 4 * 10 + o2
            ]
            ret = bytes_io.write(bytes_to_write)
            assert ret == o + 4 * 10 + o2

            # Check converted fp16 vectors
            _test_fp32_to_fp16_conversion(fp32_values[5:15], bytes_io._fp16_np[5:15])

            # Check if it has incomplete bytes correctly
            vector_idx = 4
            assert vector_idx in bytes_io._incomplete_vector_value
            assert bytes_io._incomplete_vector_value[vector_idx]["count"] == o
            j = 0  # Index of bytes_to_write
            i = 4 - o
            while i < 4:
                assert (
                    bytes_io._incomplete_vector_value[vector_idx]["bytes"][i]
                    == bytes_to_write[j]
                )
                j += 1
                i += 1

            # Check if it has incomplete bytes correctly
            vector_idx = 15
            assert vector_idx in bytes_io._incomplete_vector_value
            assert bytes_io._incomplete_vector_value[vector_idx]["count"] == o2
            j = 4 * 10 + o  # Index of bytes_to_write
            i = 0
            while i < o2:
                assert (
                    bytes_io._incomplete_vector_value[vector_idx]["bytes"][i]
                    == bytes_to_write[j]
                )
                j += 1
                i += 1
