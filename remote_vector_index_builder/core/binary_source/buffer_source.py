# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.binary_source.binary_source import BinarySource
from core.object_store.object_store import ObjectStore

from io import BytesIO
import numpy as np
from numpy.typing import NDArray


class BufferSource(BinarySource):
    def __init__(self):
        self._source = BytesIO()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._source.close()

    def read_from_object_store(
        self, object_store: ObjectStore, object_store_path: str
    ) -> None:
        object_store.read_blob(object_store_path, self._source)

    def parse(self, dtype: str) -> NDArray:
        vector_view = self._source.getbuffer()
        np_array = np.frombuffer(vector_view, dtype=dtype)
        return np_array
