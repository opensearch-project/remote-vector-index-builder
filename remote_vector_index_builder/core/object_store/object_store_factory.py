# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from typing import Any, Dict

from core.common.exceptions import UnsupportedObjectStoreTypeError
from core.common.models import IndexBuildParameters
from core.object_store.object_store import ObjectStore
from core.object_store.s3.s3_object_store import S3ObjectStore
from core.object_store.types import ObjectStoreType


class ObjectStoreFactory:
    """
    A factory class for creating object store instances.

    This class provides a static method to create appropriate object store instances
    based on the repository type specified in the index build parameters. It serves
    as a central point for object store instance creation and helps maintain loose
    coupling between different object store implementations.
    """

    @staticmethod
    def create_object_store(
        index_build_params: IndexBuildParameters, object_store_config: Dict[str, Any]
    ) -> ObjectStore:
        """
        Creates and returns an appropriate object store instance based on the repository type.

        Args:
            index_build_params (IndexBuildParameters): Parameters for index building, including
                the repository type that determines which object store implementation to use.
            object_store_config (Dict[str, Any]): Configuration dictionary containing settings
                specific to the object store implementation.

        Returns:
            ObjectStore: An instance of the appropriate object store implementation.

        Raises:
            UnsupportedObjectStoreTypeError: If the specified repository type is not supported.

        Example:
            params = IndexBuildParameters(repository_type=ObjectStoreType.S3)
            config = {"region": "us-west-2"}
            store = ObjectStoreFactory.create_object_store(params, config)
        """
        if index_build_params.repository_type == ObjectStoreType.S3:
            return S3ObjectStore(index_build_params, object_store_config)
        else:
            raise UnsupportedObjectStoreTypeError(
                f"Unknown object store type: {index_build_params.repository_type}"
            )
