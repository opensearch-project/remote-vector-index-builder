import faiss
from core.common.models import VectorsDataset, SpaceType

from core.common.models.index_builder.faiss import (
    FaissGPUIndexCagraBuilder,
    FaissIndexHNSWCagraBuilder,
)
from core.index_builder.index_builder_utils import (
    calculate_ivf_pq_n_lists,
    get_omp_num_threads,
)
from core.index_builder.interface import IndexBuildService
from timeit import default_timer as timer
import logging


class FaissIndexBuildService(IndexBuildService):
    """
    Orchestrates the workflow of
    - creating a GPU Index for the specified vectors dataset,
    - converting into CPU compatible Index
    - and writing the CPU Index to disc
    Uses the faiss library methods to achieve this.
    """

    def __init__(self):
        self.omp_num_threads = get_omp_num_threads()

    def build_index(
        self,
        gpu_build_params: dict,
        cpu_build_params: dict,
        vectors_dataset: VectorsDataset,
        workloadToExecute: dict,
        cpu_index_output_file_path: str,
    ):
        faiss_gpu_index_cagra_builder = None
        faiss_index_hnsw_cagra_builder = None
        faiss_gpu_build_index_output = None
        faiss_cpu_build_index_output = None

        try:
            # Set number of threads for parallel processing
            logging.info("In build index")
            faiss.omp_set_num_threads(self.omp_num_threads)

            space_type = (
                SpaceType.L2
                if workloadToExecute.get("space-type") is None
                else SpaceType(workloadToExecute.get("space-type"))
            )

            gpu_build_params["ivf_pq_params"]["n_lists"] = calculate_ivf_pq_n_lists(
                len(vectors_dataset.vectors)
            )

            # Step 1a: Initialize GPU Config
            faiss_gpu_index_cagra_builder = FaissGPUIndexCagraBuilder.from_dict(
                gpu_build_params
            )

            # Step 1b: Create a GPU Index from the faiss config and vector dataset
            start = timer()
            faiss_gpu_build_index_output = (
                faiss_gpu_index_cagra_builder.build_gpu_index(
                    vectors_dataset, workloadToExecute["dimension"], space_type
                )
            )
            gpu_index_build_time = timer() - start

            # Step 2a: Initialize CPU Config
            faiss_index_hnsw_cagra_builder = FaissIndexHNSWCagraBuilder.from_dict(
                cpu_build_params
            )

            # Step 2b: Convert GPU Index to CPU Index, update index to cpu index in index-id mappings
            # Also Delete GPU Index after conversion
            start = timer()
            faiss_cpu_build_index_output = (
                faiss_index_hnsw_cagra_builder.convert_gpu_to_cpu_index(
                    faiss_gpu_build_index_output
                )
            )
            gpu_to_cpu_conversion_time = timer() - start

            # Step 3: Write CPU Index to persistent storage
            start = timer()
            faiss_index_hnsw_cagra_builder.write_cpu_index(
                faiss_cpu_build_index_output, cpu_index_output_file_path
            )
            write_cpu_index_time = timer() - start

            del gpu_build_params["ivf_pq_params"]["n_lists"]
            return {
                "indexTime": gpu_index_build_time,
                "writeIndexTime": write_cpu_index_time,
                "totalTime": gpu_index_build_time
                + gpu_to_cpu_conversion_time
                + write_cpu_index_time,
                "unit": "seconds",
                "gpu_to_cpu_index_conversion_time": gpu_to_cpu_conversion_time,
                "write_to_file_time": write_cpu_index_time,
            }

        except Exception as exception:
            logging.info("There was an exception: " + str(exception))
            # Clean up GPU Index Response if orchestrator failed after GPU Index Creation
            if faiss_gpu_build_index_output is not None:
                try:
                    faiss_gpu_build_index_output.cleanup()
                except Exception as e:
                    print(f"Warning: Failed to clean up GPU index response: {str(e)}")

            # Clean up CPU Index Response if orchestrator failed after CPU Index Creation
            if faiss_cpu_build_index_output is not None:
                try:
                    faiss_cpu_build_index_output.cleanup()
                except Exception as e:
                    print(f"Warning: Failed to clean up CPU index response: {str(e)}")
            raise Exception(
                f"Faiss Index Build Service build_index workflow failed. Reason: {str(exception)}"
            ) from exception
