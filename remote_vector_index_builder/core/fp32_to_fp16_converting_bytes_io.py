import os
from io import BytesIO

import numpy as np
import threading

"""
This class converts raw FP32 byte streams into FP16 values and stores them in a pre-allocated FP16 NumPy array. It is
designed to handle input arriving in multiple partitions from a continuous FP32 vector stream.
Because a partition boundary may split an FP32 value across chunks, the class first identifies all complete FP32 values
from the incoming bytes, converts them to FP16, and writes them into the FP16 array. Any remaining incomplete bytes are
stored in a dictionary and reassembled once the rest of the bytes arrive in future partitions.

For example, let's say we had 6 FP32 values as below:
[[b0, b1, b2, b3], [b4, b5, b6, b7], [b8, b9, b10, b11], [b12, b13, b14, b15], [b16, b17, b18, b19],
 [b20, b21, b22, b23]]
where [b0, b1, b2, b3] represents four bytes for the first FP32 value.

The FP32 data can be split across two partitions as shown below:
Partition-1:
[[b0, b1, b2, b3], [b4, b5, b6, b7], [b8, b9, b10, b11], [b12, b13]]

Partition-2:
[[b14, b15], [b16, b17, b18, b19], [b20, b21, b22, b23]]

Assume partition-1 arrives before partition-2.
This class first identifies complete FP32 values (each consisting of 4 bytes). From partition-1, the following groups
are detected:

    [[b0, b1, b2, b3], [b4, b5, b6, b7], [b8, b9, b10, b11]]

These complete 4-byte chunks are immediately converted to FP16 values and stored in the FP16 NumPy array. Any remaining
incomplete bytes — in this case [b12, b13] — are stored in a dictionary, keyed by the ordinal index of the FP32 vector,
which is 3 here. (e.g. represents incomplete bytes for 3rd vector)

When partition-2 arrives, the same logic is applied. It identifies complete FP32 values:

    [[b16, b17, b18, b19], [b20, b21, b22, b23]]. These values will be put into FP16 numpy array.

These are likewise converted and stored. Meanwhile, [b14, b15] are also recognized as belonging to the previously
incomplete FP32 value at index 3. Now that all 4 bytes for that vector are available ([b12, b13, b14, b15]),
they are reassembled, converted to FP16, and written to the FP16 array. The corresponding entry in the dictionary is
then removed.
"""


class FP32ToFP16ConvertingBytesIO(BytesIO):
    def __init__(self, num_floats):
        BytesIO.__init__(self)
        self._fp16_np = np.zeros(num_floats, dtype=np.float16)
        self._curr_offset = 0
        self._incomplete_vector_value = dict()
        self._lock = threading.Lock()

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        with self._lock:
            if whence == os.SEEK_SET:
                self._curr_offset = offset
            elif whence == os.SEEK_CUR:
                self._curr_offset += offset
            elif whence == os.SEEK_END:
                self._curr_offset = np.dtype(np.float32).itemsize * len(self._fp16_np)
            else:
                raise ValueError(f"Unexpected whence={whence}")

    def getbuffer(self):
        if len(self._incomplete_vector_value) != 0:
            raise RuntimeError(
                f"There're still {len(self._incomplete_vector_value)} incomplete FP32 byte values"
            )
        return memoryview(self._fp16_np)

    def write(self, b):
        with self._lock:
            len_bytes = len(b)
            # Determine the boundary of the first float value
            # if byte_idx1 == 0, meaning the offset is located at the multiple of sizeof(float), the start offset of
            # float value. Otherwise, it is pointing to incomplete bytes within one float value.
            # For example, value_idx1=55, byte_idx1=2 then the offset is pointing to b2 in
            # [...54 float values, [?, ?, b2, b3]]
            head_value_idx, head_byte_idx = FP32ToFP16ConvertingBytesIO._get_index(
                self._curr_offset
            )

            # We skip incomplete float value for now when having non-zero byte_idx1
            # Otherwise, we can use the given value_idx1
            copy_start_index = (
                head_value_idx if head_byte_idx == 0 else head_value_idx + 1
            )

            # Determine the boundary of the last float value
            # if byte_idx2 == 0, the offset is located at the multiple of sizeof(float), the start offset of
            # float value. Otherwise, it is pointing to incomplete bytes within one float value.
            # For example, value_idx2=55, byte_idx2=2 then the offset is pointing to b2 in
            # [...54 float values, [?, ?, b2, b3]]
            actual_end_offset = self._curr_offset + len_bytes
            tail_value_idx, tail_byte_idx = FP32ToFP16ConvertingBytesIO._get_index(
                actual_end_offset
            )
            copy_end_index = tail_value_idx

            # Clip bytes to have complete float values
            clip_start = 0 if head_byte_idx == 0 else 4 - head_byte_idx
            clip_end = len_bytes - tail_byte_idx
            fp32_vector_values = np.frombuffer(b[clip_start:clip_end], dtype=np.float32)

            # Convert FP32 values to FP16
            self._fp16_np[copy_start_index:copy_end_index] = fp32_vector_values

            # Try to assemble incomplete float value from leading and trailing
            self._append_incomplete_bytes(
                head_value_idx, b, 0, clip_start, head_byte_idx
            )
            self._append_incomplete_bytes(tail_value_idx, b, clip_end, len_bytes, 0)

            self._curr_offset += len_bytes
            return len_bytes

    @staticmethod
    def _get_index(offset):
        # sizeof(fp32) == 4
        size_of_fp32 = np.dtype(np.float32).itemsize
        return int(offset / size_of_fp32), int(offset % size_of_fp32)

    def _append_incomplete_bytes(
        self, value_idx, buffer, start_offset, end_offset, byte_idx
    ):
        if start_offset == end_offset:
            return

        bytes_count = self._incomplete_vector_value.get(value_idx)
        if bytes_count is None:
            bytes_count = {"count": 0, "bytes": [0] * 4}
            self._incomplete_vector_value[value_idx] = bytes_count
        four_bytes = bytes_count["bytes"]

        offset = start_offset
        while offset < end_offset:
            four_bytes[byte_idx] = buffer[offset]
            offset += 1
            byte_idx += 1

        bytes_count["count"] += end_offset - start_offset
        if bytes_count["count"] == 4:
            self._fp16_np[value_idx] = np.frombuffer(
                bytes(four_bytes), dtype=np.float32
            )[0]
            del self._incomplete_vector_value[value_idx]
