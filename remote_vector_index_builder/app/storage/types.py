# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from enum import Enum


class RequestStoreType(str, Enum):
    """
    Enumeration of supported request store types.

    This enum defines the available storage backend types for the request store system.
    Inherits from str and Enum to provide string-based enumeration values.

    Attributes:
        MEMORY (str): In-memory storage backend, data is stored in application memory
    """

    MEMORY = "memory"
