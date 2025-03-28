# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from app.base.config import Settings
from app.storage.base import RequestStore
from app.storage.memory import InMemoryRequestStore
from app.storage.types import RequestStoreType


class RequestStoreFactory:
    """
    Factory class for creating RequestStore instances.

    This class provides a static factory method to create different types of
    RequestStore implementations based on the specified store type.
    """

    @staticmethod
    def create(store_type: RequestStoreType, settings: Settings) -> RequestStore:
        """
        Creates and returns a RequestStore instance based on the specified type.

        Args:
            store_type (RequestStoreType): The type of request store to create
            settings (Settings): Configuration settings for the request store

        Returns:
            RequestStore: An instance of the specified RequestStore implementation

        Raises:
            ValueError: If the specified store type is not supported
        """
        if store_type == RequestStoreType.MEMORY:
            return InMemoryRequestStore(settings)
        else:
            raise ValueError(f"Unsupported request store type: {store_type}")
