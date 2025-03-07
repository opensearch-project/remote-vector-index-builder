from dataclasses import dataclass
import faiss


@dataclass
class CreateGPUIndexResponse:
    """
    A data class that holds FAISS GPU index components for vector similarity search.

    This class manages two types of FAISS indexes:
    1. A GPU-based CAGRA index for efficient similarity search
    2. An ID mapping index to maintain relationships between vectors and their identifiers

    Attributes:
        gpu_index_cagra (faiss.GpuIndexCagra): The GPU-accelerated CAGRA index for
            performing vector similarity searches.
        id_map_index (faiss.Index): An index that maintains mapping between vectors
            and their corresponding identifiers.
    """

    gpu_index_cagra: faiss.GpuIndexCagra
    id_map_index: faiss.Index

    def __del__(self):
        """
        Destructor to clean up FAISS resources.
        Ensures GPU memory is properly freed when the object is destroyed.

        The method handles cleanup by
        explicitly deleting both indexes to free GPU and system memory
        """

        try:
            if self.gpu_index_cagra:
                self.gpu_index_cagra.thisown = True
                del self.gpu_index_cagra
            if self.id_map_index:
                self.id_map_index.own_fields = True
                del self.id_map_index
        except Exception as e:
            print(f"Error during cleanup: {e}")
