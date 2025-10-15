# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import logging
import os
from typing import Optional, Tuple
from app.models.workflow import BuildWorkflow
from core.object_store.s3.s3_object_store_config import S3ClientConfig
from core.common.models import IndexSerializationMode
from core.tasks import run_tasks

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Handles the building of indexes based on provided workflows.
    """

    def build_index(
        self, workflow: BuildWorkflow
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Builds the index for the given workflow, using the run_tasks function

        Args:
            workflow (BuildWorkflow): Workflow containing index build parameters

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: A tuple containing:
                - Success status (True/False)
                - Index path if successful, None otherwise
                - Error message if failed, None otherwise
        """
        s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL", None)
        index_serialization_mode = os.environ.get(
            "INDEX_SERIALIZATION_MODE", IndexSerializationMode.DISK
        )
        result = run_tasks(
            workflow.index_build_parameters,
            {
                "s3_client_config": S3ClientConfig(
                    region_name=os.environ.get("AWS_DEFAULT_REGION", None),
                    endpoint_url=s3_endpoint_url,
                ),
            },
            index_serialization_mode,
        )
        if not result.file_name:
            return False, None, result.error
        return True, result.file_name, None
