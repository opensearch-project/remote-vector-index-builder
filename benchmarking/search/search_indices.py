import os.path

import faiss
import sys
import logging

from timeit import default_timer as timer
from benchmarking.utils.common_utils import recall_at_r
import numpy as np
from tqdm import tqdm


def runIndicesSearch(xq, graphFile: str, param: dict, gt) -> dict:
    index: faiss.IndexIDMap = loadGraphFromFile(graphFile)
    index.index.base_level_only = True
    hnswParameters = faiss.SearchParametersHNSW()
    hnswParameters.efSearch = (
        100 if param.get("ef_search") is None else param["ef_search"]
    )
    logging.info(f"Ef search is : {hnswParameters.efSearch}")
    k = 100 if param.get("K") is None else param["K"]

    def search(xq, k, params):
        D, ids = index.search(xq, k, params=params)
        return ids

    # K is always set to 100
    total_time = 0
    I = []
    for query in tqdm(
        xq,
        total=len(xq),
        desc=f"Running queries for ef_search: {hnswParameters.efSearch}",
    ):
        t1 = timer()
        result = search(np.array([query]), 100, hnswParameters)
        t2 = timer()
        I.append(result[0])
        total_time = total_time + (t2 - t1)

    recall_at_k = recall_at_r(I, gt, k, k, len(xq))
    recall_at_1 = recall_at_r(I, gt, 1, 1, len(xq))
    logging.info(f"Recall at {k} : is {recall_at_k}")
    logging.info(f"Recall at 1 : is {recall_at_1}")
    # deleting the index to avoid OOM
    # We don't need to set own_fileds = true as this will be automatically set by faiss while reading the index.
    del index
    return {
        "searchTime": total_time,
        "units": "seconds",
        f"recall_at_{k}": recall_at_k,
        "recall_at_1": recall_at_1,
        "total_queries": len(xq),
        "search_throughput": len(xq) / total_time,
    }


def loadGraphFromFile(graphFile: str) -> faiss.Index:
    if os.path.isfile(graphFile) is False:
        logging.error(f"The path provided: {graphFile} is not a file")
        sys.exit(0)

    return faiss.read_index(graphFile)
