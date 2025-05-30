# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from io import BytesIO

print(dir())

from core.object_store.s3.s3_object_store import S3ObjectStore
from core.common.models import IndexBuildParameters
from core.object_store.types import ObjectStoreType


def get_encrypted_object_and_reupload():
    index_build_params = IndexBuildParameters(
        vector_path="cohere_1k_vectors.knnvec",  # Will be set per dataset
        doc_id_path="cohere_1k_docids.knndid",  # Will be set per dataset
        repository_type=ObjectStoreType.S3,
        container_name="<bucket-name>",
        dimension=128,  # Default dimension, will be overridden per dataset
        doc_count=5,  # Will be set per dataset
    )
    object_store_config = {}
    object_store = S3ObjectStore(index_build_params, object_store_config)

    buffer = BytesIO()
    object_store.read_blob(index_build_params.vector_path, buffer)
    print(object_store.upload_args)
    object_store.write_blob(
        "resources/cohere_1k_vectors",
        "cohere_1k_vectors_reuploaded",
    )


if __name__ == "__main__":
    print("Manually running")
    get_encrypted_object_and_reupload()
