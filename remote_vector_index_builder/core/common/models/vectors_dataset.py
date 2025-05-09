# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass

import numpy as np


@dataclass
class VectorsDataset:
    """A class for handling vector datasets and their associated document IDs.

    Attributes:
        vectors (numpy.ndarray): The array of vectors, where each row represents a vector.
        doc_ids (numpy.ndarray): Array of document IDs corresponding to the vectors.
    """

    vectors: np.ndarray
    doc_ids: np.ndarray

    def free_vectors_space(self):
        """Free up memory by deleting the vectors and document IDs arrays."""

        try:
            del self.vectors
        except AttributeError:
            pass
        try:
            del self.doc_ids
        except AttributeError:
            pass
