# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from abc import ABC, abstractmethod

from core.object_store.object_store import ObjectStore
from numpy.typing import NDArray


class BinarySource(ABC):

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def read_from_object_store(
        self, object_store: ObjectStore, object_store_path: str
    ) -> None:
        pass

    @abstractmethod
    def parse(self, dtype: str) -> NDArray:
        pass

    def transform_to_numpy_array(
        self,
        object_store: ObjectStore,
        object_store_path: str,
        dtype: str,
        expected_length: int,
    ) -> NDArray:
        self.read_from_object_store(object_store, object_store_path)
        np_array = self.parse(dtype)
        if len(np_array) != expected_length:
            raise ValueError(
                f"Expected {expected_length} vectors, but got {len(np_array)}"
            )
        return np_array
