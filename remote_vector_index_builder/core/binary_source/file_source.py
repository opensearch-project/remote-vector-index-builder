# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.binary_source.binary_source import BinarySource
from core.object_store.object_store import ObjectStore

from tempfile import TemporaryDirectory
import numpy as np
from numpy.typing import NDArray


class FileSource(BinarySource):
    def __init__(self):
        self._temp_dir = None
        self._source = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._source is not None:
            self._source.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def read_from_object_store(
        self, object_store: ObjectStore, object_store_path: str
    ) -> None:
        filename = object_store_path.replace("/", "_").replace("\\", "_")
        self._temp_dir = TemporaryDirectory()
        self._source = open(f"{self._temp_dir.name}/{filename}", "wb+")
        object_store.read_blob(object_store_path, self._source)

    def parse(self, dtype: str) -> NDArray:
        np_array = np.memmap(self._source, dtype=dtype, mode="r+")
        return np_array
