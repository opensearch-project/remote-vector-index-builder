# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
from unittest.mock import Mock
from app.storage.factory import RequestStoreFactory
from app.storage.types import RequestStoreType
from app.storage.memory import InMemoryRequestStore


class TestRequestStoreFactory:
    def setup_method(self):
        """Setup common test fixtures"""
        self.mock_settings = Mock()

    def test_create_memory_store(self):
        """Test creation of in-memory store"""
        store = RequestStoreFactory.create(RequestStoreType.MEMORY, self.mock_settings)
        assert isinstance(store, InMemoryRequestStore)

    def test_unsupported_store_type(self):
        """Test creating store with unsupported type raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            RequestStoreFactory.create("INVALID_TYPE", self.mock_settings)
        assert "Unsupported request store type" in str(exc_info.value)

    def test_none_store_type(self):
        """Test creating store with None type raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            RequestStoreFactory.create(None, self.mock_settings)
        assert "Unsupported request store type" in str(exc_info.value)
