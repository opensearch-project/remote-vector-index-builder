# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.common.models.index_build_parameters import IndexBuildParameters
import time
from core import create_vectors_dataset, upload_index
import logging
from io import BytesIO


def configure_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    index_build_params = {
        "repository_type": "s3",
        "container_name": "testbucket-rchital",
        "vector_path": "target_field__53.knnvec",
        "doc_id_path": "target_field__53.knndid",
        "dimension": 1000,
        "doc_count": 76800,
    }

    vector_bytes_buffer = BytesIO()
    doc_id_bytes_buffer = BytesIO()
    model = IndexBuildParameters.model_validate(index_build_params)
    start_time = time.time()
    create_vectors_dataset(
        model, {"debug": True}, vector_bytes_buffer, doc_id_bytes_buffer
    )
    end_time = time.time()
    print(end_time - start_time)

    # start_time = time.time()
    # upload_index(model, {'debug': True}, "./target_field__53.knnvec")
    # end_time = time.time()
    # print(end_time - start_time)


if __name__ == "__main__":
    configure_logging("INFO")
    main()
