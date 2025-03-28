# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from typing import Any


def get_field_path(location: tuple[Any, ...]) -> str:
    """Convert a location tuple to a readable field path string.
    This function is used to format errors thrown by
    Pydantic input validation.

    Args:
        location (tuple[Any, ...]): A tuple containing path elements, which can be
            either integers (for array indices) or strings (for field names).

    Returns:
        str: A formatted string representing the field path where:
            - Integer elements are formatted as array indices (e.g., "[0]")
            - String elements are joined with dots (e.g., "field.subfield")
            - First string element appears without a leading dot

    """
    field_path = []
    for loc in location:
        if isinstance(loc, int):
            field_path.append(f"[{loc}]")
        elif isinstance(loc, str):
            if field_path:
                field_path.append("." + loc)
            else:
                field_path.append(loc)
    return "".join(field_path)
