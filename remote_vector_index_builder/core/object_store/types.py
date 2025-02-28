# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from enum import Enum


class ObjectStoreType(str, Enum):
    """
    Enumeration of supported object store types.

    This enum inherits from both str and Enum to provide string-based
    enumeration values, allowing for easy serialization and comparison.

    Attributes:
        S3 (str): Represents Amazon S3 object storage service
    """

    S3 = "s3"
