# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import pytest
from core.common.exceptions import UnsupportedObjectStoreTypeError
from core.object_store.object_store_factory import ObjectStoreFactory
from core.object_store.s3.s3_object_store import S3ObjectStore
from core.object_store.types import ObjectStoreType


def test_create_s3_object_store(index_build_parameters, object_store_config):
    """Test creating an S3 object store"""
    # Setup
    index_build_parameters.repository_type = ObjectStoreType.S3

    # Execute
    store = ObjectStoreFactory.create_object_store(
        index_build_params=index_build_parameters,
        object_store_config=object_store_config,
    )

    # Assert
    assert isinstance(store, S3ObjectStore)


def test_create_object_store_unsupported_type(
    index_build_parameters, object_store_config
):
    """Test creating an object store with unsupported type raises error"""
    # Setup
    index_build_parameters.repository_type = "unsupported_type"

    # Execute and Assert
    with pytest.raises(UnsupportedObjectStoreTypeError):
        ObjectStoreFactory.create_object_store(
            index_build_params=index_build_parameters,
            object_store_config=object_store_config,
        )
