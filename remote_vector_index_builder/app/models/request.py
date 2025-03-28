# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from pydantic import BaseModel


class RequestParameters(BaseModel):
    """
    Model representing the parameters required for a vector index build request.

    This class validates and stores the essential parameters needed to process
    a vector index building request, including the path to the vector data and
    the tenant identification.

    Attributes:
        vector_path (str): Path to the vector data file or resource
        tenant_id (str): Unique identifier for the tenant making the request
    """

    vector_path: str
    tenant_id: str

    def __str__(self):
        """
        Create a string representation of the request parameters.

        Returns:
            str: A string in the format "{vector_path}-{tenant_id}"
        """
        return f"{self.vector_path}-{self.tenant_id}"

    def __eq__(self, other):
        """
        Compare this RequestParameters instance with another object.

        Args:
            other: The object to compare with this instance

        Returns:
            bool: True if the other object is a RequestParameters instance
                 with the same string representation, False otherwise
        """
        if not isinstance(other, RequestParameters):
            return False
        return str(self) == str(other)
