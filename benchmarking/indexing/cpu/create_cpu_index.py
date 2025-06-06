import logging

import faiss
from benchmarking.decorators.timer import timer_func
from benchmarking.utils.common_utils import get_omp_num_threads
from timeit import default_timer as timer


def indexData(
    d, xb, ids, param, space_type, file_to_write="cpuIndex.hnsw.graph"
) -> dict:
    num_of_parallel_threads = get_omp_num_threads()
    logging.info(
        f"Setting number of parallel threads for graph build: {num_of_parallel_threads}"
    )
    faiss.omp_set_num_threads(num_of_parallel_threads)

    m = 16 if param.get("m") is None else param.get("m")

    metric = faiss.METRIC_L2
    if space_type == "innerproduct":
        metric = faiss.METRIC_INNER_PRODUCT
    cpuPureHNSWIndex: faiss.IndexHNSWFlat = faiss.index_factory(
        d, f"HNSW{m},Flat", metric
    )

    cpuPureHNSWIndex.hnsw.efConstruction = (
        512 if param.get("ef_construction") is None else param.get("ef_construction")
    )
    logging.info(
        f"EF Construction is : {cpuPureHNSWIndex.hnsw.efConstruction} and m is : {m}"
    )
    cpuIdMapIndex = faiss.IndexIDMap(cpuPureHNSWIndex)

    @timer_func
    def indexDataInIndex(index: faiss.Index, ids, xb):
        index.add_with_ids(xb, ids)

    t1 = timer()
    indexDataInIndex(cpuIdMapIndex, ids, xb)
    t2 = timer()
    indexTime = t2 - t1

    @timer_func
    def writeIndex(index, fileName):
        faiss.write_index(index, fileName)

    t1 = timer()
    writeIndex(cpuIdMapIndex, file_to_write)
    t2 = timer()
    writeIndexTime = t2 - t1
    # this will ensure that everything is deleted
    cpuPureHNSWIndex.own_fields = True
    del cpuPureHNSWIndex
    del cpuIdMapIndex
    return {
        "indexTime": indexTime,
        "writeIndexTime": writeIndexTime,
        "totalTime": indexTime + writeIndexTime,
        "unit": "seconds",
    }
