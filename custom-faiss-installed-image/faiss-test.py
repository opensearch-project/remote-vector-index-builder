import sys
import os.path
# If you faiss python installed you should skip this line
sys.path.append('/tmp/faiss/build/faiss/python')
import faiss
import logging
from timeit import default_timer as timer
import math
import numpy as np
import time

from numpy import dtype


def indexData(d:int, xb:np.ndarray, ids:np.ndarray, indexingParams:dict, space_type:str, file_to_write:str="gpuIndex.cagra.graph"):
    num_of_parallel_threads = 8
    logging.info(f"Setting number of parallel threads for graph build: {num_of_parallel_threads}")
    faiss.omp_set_num_threads(num_of_parallel_threads)
    res = faiss.StandardGpuResources()
    metric = faiss.METRIC_L2
    if space_type == "innerproduct":
        metric = faiss.METRIC_INNER_PRODUCT
    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 64 if indexingParams.get('intermediate_graph_degree') is None else indexingParams['intermediate_graph_degree']
    cagraIndexConfig.graph_degree = 32 if indexingParams.get('graph_degree') == None else indexingParams['graph_degree']
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    #cagraIndexConfig.conservative_memory_allocation = True
    # This was available earlier now this parameter is now available
    #cagraIndexConfig.store_dataset = False

    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
    cagraIndexIVFPQConfig.kmeans_n_iters = 10 if indexingParams.get('kmeans_n_iters') == None else indexingParams['kmeans_n_iters']
    cagraIndexIVFPQConfig.pq_bits = 8 if indexingParams.get('pq_bits') == None else indexingParams['pq_bits']
    cagraIndexIVFPQConfig.pq_dim = 32 if indexingParams.get('pq_dim') == None else indexingParams['pq_dim']
    cagraIndexIVFPQConfig.n_lists = int(math.sqrt(len(xb))) if indexingParams.get('n_lists') == None else indexingParams['n_lists']
    cagraIndexIVFPQConfig.kmeans_trainset_fraction = 10 if indexingParams.get('kmeans_trainset_fraction') == None else indexingParams['kmeans_trainset_fraction']
    cagraIndexIVFPQConfig.conservative_memory_allocation = True
    cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig

    cagraIndexSearchIVFPQConfig = faiss.IVFPQSearchCagraConfig()
    cagraIndexSearchIVFPQConfig.n_probes = 30 if indexingParams.get('n_probes') == None else indexingParams['n_probes']
    cagraIndexConfig.ivf_pq_search_params = cagraIndexSearchIVFPQConfig

    print("Creating GPU Index.. with IVF_PQ")
    t1 = timer()
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)
    indexDataInIndex(idMapIVFPQIndex, ids, xb)
    t2 = timer()
    print(f"Indexing time: {t2 - t1} sec")

def indexDataInIndex(index: faiss.Index, ids, xb):
    if ids is None:
        index.train(xb)
    else:
        index.add_with_ids(xb, ids)

def runIndicesSearch(xq, graphFile:str, param:dict, gt) -> dict:
    index:faiss.IndexIDMap = loadGraphFromFile(graphFile)
    index.index.base_level_only = True
    hnswParameters = faiss.SearchParametersHNSW()
    hnswParameters.efSearch = 100 if param.get('ef_search') is None else param['ef_search']
    logging.info(f"Ef search is : {hnswParameters.efSearch}")
    k = 100 if param.get('K') is None else param['K']
    
    def search(xq, k, params):
        D, ids = index.search(xq, k, params=params)
        return ids
    # K is always set to 100
    total_time = 0
    I = []
    query = xq[0]
    t1 = timer()
    result = search(np.array([query]), 100, hnswParameters)
    t2 = timer()
    I.append(result[0])
    total_time = total_time + (t2-t1)
    print(f"Total time for search: {total_time}")



def loadGraphFromFile(graphFile: str) -> faiss.Index:
    if os.path.isfile(graphFile) is False:
        logging.error(f"The path provided: {graphFile} is not a file")
        sys.exit(0)

    return faiss.read_index(graphFile)



if __name__ == "__main__":
    d = 768
    filename = 'mmap-7m.npy'
    print("file is written, loading file now..")
    xb = np.load(file = filename)
    print(xb.shape)

    #ids = [i for i in range(len(xb))]
    ids = None
    indexData(d, xb, ids, {}, "l2", "testgpuIndex.cagra.graph")
    print("Indexing done")
    del xb
    time.sleep(10)
    print("Deleted xb")
