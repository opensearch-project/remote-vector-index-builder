import yaml
import logging
import sys
import os
from typing import List

from benchmarking.data_types.data_types import IndexTypes, WorkloadTypes
from benchmarking.dataset import dataset_utils
from benchmarking.memory_profiler.memory_monitor import MemoryMonitor
from benchmarking.search import search_indices
from benchmarking.utils.common_utils import ensureDir
import json
import time
from tqdm import tqdm
import copy


from core.common.models import VectorsDataset, SpaceType
from benchmarking.service.faiss_index_build_service import FaissIndexBuildService


def runWorkload(
    workloadNames: List[str], indexTypeStr: str, workloadType: WorkloadTypes
):
    allWorkloads = readAllWorkloads()
    indexTypesList = []
    if indexTypeStr == "all":
        indexTypesList = IndexTypes.enumList()
    else:
        indexTypesList.append(IndexTypes.from_str(indexTypeStr))

    for indexType in indexTypesList:
        # if workloadNames are empty, default to running all workloads
        if len(workloadNames) == 0:
            for currentWorkloadName in allWorkloads[indexType.value]:
                executeWorkload(
                    workloadName=currentWorkloadName,
                    workloadToExecute=allWorkloads[indexType.value][
                        currentWorkloadName
                    ],
                    indexType=indexType,
                    workloadType=workloadType,
                )
        else:
            for workloadName in workloadNames:
                executeWorkload(
                    workloadName=workloadName,
                    workloadToExecute=allWorkloads[indexType.value][workloadName],
                    indexType=indexType,
                    workloadType=workloadType,
                )


def executeWorkload(
    workloadName: str,
    workloadToExecute: dict,
    indexType: IndexTypes,
    workloadType: WorkloadTypes,
):
    workloadToExecute["indexType"] = indexType.value
    logging.info(workloadToExecute)
    dataset_file = dataset_utils.downloadDataSetForWorkload(workloadToExecute)
    allMetrics = {f"{workloadName}": {}}
    if (
        workloadType == WorkloadTypes.INDEX_AND_SEARCH
        or workloadType == WorkloadTypes.INDEX
    ):
        indexingMetrics = doIndexing(workloadToExecute, dataset_file, indexType)
        allMetrics[workloadName] = {
            "workload-details": indexingMetrics["workload-details"],
            "indexingMetrics": indexingMetrics["indexing-metrics"],
        }

    if (
        workloadType == WorkloadTypes.INDEX_AND_SEARCH
        or workloadType == WorkloadTypes.SEARCH
    ):
        searchMetrics = doSearch(workloadToExecute, dataset_file, indexType)
        allMetrics[workloadName]["searchMetrics"] = searchMetrics["search-metrics"]
        allMetrics[workloadName]["workload-details"] = searchMetrics["workload-details"]

    logging.info(json.dumps(allMetrics))
    persistMetricsAsJson(workloadType, allMetrics, workloadName, indexType)


def persistMetricsAsJson(
    workloadType: WorkloadTypes,
    allMetrics: dict,
    workloadName: str,
    indexType: IndexTypes,
):
    dir_path = ensureDir(f"results/{workloadName}")
    with open(f"{dir_path}/{workloadType.value}_{indexType.value}.json", "w") as file:
        json.dump(allMetrics, file, indent=4)


def doIndexing(workloadToExecute: dict, datasetFile: str, indexType: IndexTypes):
    logging.info("Run Indexing...")
    d, xb, ids = dataset_utils.prepare_indexing_dataset(
        datasetFile,
        workloadToExecute.get("normalize"),
        workloadToExecute.get("indexing-docs"),
    )

    workloadToExecute["dimension"] = d
    workloadToExecute["vectorsCount"] = len(xb)

    parameters_level_metrics = []
    for param in tqdm(workloadToExecute["indexing-parameters"]):
        # TODO: Include compression in indexing-parameters field in yaml
        # If compression is in indexing-parameters field, then we don't need this if statement and inner for loop
        if indexType == IndexTypes.GPU:
            for compression in workloadToExecute["compression"]:
                param = copy.deepcopy(param)
                compression = int(compression)
                if compression != 0:
                    param["ivf_pq_params"]["pq_dim"] = int(
                        workloadToExecute["dimension"] / compression
                    )
                else:
                    param["ivf_pq_params"]["pq_dim"] = 0
                metrics = get_indexing_metrics(
                    workloadToExecute, indexType, param, xb, ids
                )
                parameters_level_metrics.append(metrics)
        else:
            metrics = get_indexing_metrics(workloadToExecute, indexType, param, xb, ids)
            parameters_level_metrics.append(metrics)

        logging.info("Sleeping for 5 sec for better metrics capturing")
        time.sleep(5)

    del xb
    del ids
    return {
        "workload-details": workloadToExecute,
        "indexing-metrics": parameters_level_metrics,
    }


def doSearch(workloadToExecute: dict, datasetFile: str, indexType: IndexTypes):
    logging.info("Running Search...")
    d, xq, gt = dataset_utils.prepare_search_dataset(
        datasetFile, workloadToExecute.get("normalize")
    )
    workloadToExecute["dimension"] = d
    workloadToExecute["queriesCount"] = len(xq)
    parameters_level_metrics = []
    for indexingParam in workloadToExecute["indexing-parameters"]:
        # TODO: Include compression in indexing-parameters field in yaml
        # If compression is in indexing-parameters field, then we don't need this if statement and inner for loop
        if indexType == IndexTypes.GPU:
            for compression in workloadToExecute["compression"]:
                indexingParam = copy.deepcopy(indexingParam)
                if compression != 0:
                    indexingParam["ivf_pq_params"]["pq_dim"] = int(
                        workloadToExecute["dimension"] / compression
                    )
                else:
                    indexingParam["ivf_pq_params"]["pq_dim"] = 0
                metrics = get_search_metrics(
                    workloadToExecute, indexType, indexingParam, xq, gt
                )
                parameters_level_metrics = parameters_level_metrics + metrics
        else:
            metrics = get_search_metrics(
                workloadToExecute, indexType, indexingParam, xq, gt
            )
            parameters_level_metrics = parameters_level_metrics + metrics

    del xq
    del gt
    return {
        "workload-details": workloadToExecute,
        "search-metrics": parameters_level_metrics,
    }


def readAllWorkloads():
    with open("/benchmarking/benchmarks.yml") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit()


def get_graph_file(workloadToExecute: dict, indexType: IndexTypes, param: dict):
    dir_path = ensureDir("graphs")
    d = workloadToExecute["dimension"]
    str_to_build = f"{workloadToExecute['dataset_name']}_{d}.{indexType.value}"
    sorted_param_keys = sorted(param.keys())
    for key in sorted_param_keys:
        value = str(param[key])
        # replace special characters from param, to make graph name more readable
        special_chars_to_remove = " {}',"
        for character in special_chars_to_remove:
            value = value.replace(character, "")
        value = value.replace(":", "_")
        str_to_build += f"_{key}_{value}"
    str_to_build += ".graph"
    return os.path.join(dir_path, str_to_build)


def get_indexing_metrics(workloadToExecute, indexType, indexingParam, xb, ids):
    graph_file = get_graph_file(workloadToExecute, indexType, indexingParam)
    if os.path.exists(graph_file):
        logging.info(f"Removing file : {graph_file}")
        os.remove(graph_file)

    metrics = {"indexing-param": indexingParam}
    logging.info(
        f"================ Running configuration: {indexingParam} ================"
    )
    monitor = None

    try:
        timingMetrics = None
        if indexType == IndexTypes.GPU:
            monitor = MemoryMonitor(graph_file, monitor_gpu=True)
            monitor.start_monitoring()
            vectors_dataset = VectorsDataset(xb, ids)
            faiss_index_build_service = FaissIndexBuildService()
            timingMetrics = faiss_index_build_service.build_index(
                indexingParam,
                workloadToExecute["search-parameters"][0],
                vectors_dataset,
                workloadToExecute,
                graph_file,
            )
            del vectors_dataset
        else:
            monitor = MemoryMonitor(graph_file, monitor_gpu=False)
            monitor.start_monitoring()
            from benchmarking.indexing.cpu.create_cpu_index import (
                indexData as indexDataInCpu,
            )

            space_type = (
                SpaceType("l2")
                if workloadToExecute.get("space-type") is None
                else SpaceType(workloadToExecute.get("space-type"))
            )

            timingMetrics = indexDataInCpu(
                workloadToExecute["dimension"],
                xb,
                ids,
                indexingParam,
                space_type,
                file_to_write=graph_file,
            )

        metrics["indexing-timingMetrics"] = timingMetrics
        logging.info(f"===== Timing Metrics : {timingMetrics} ====")
        logging.info(
            f"================ Completed configuration: {indexingParam} ================"
        )
    finally:
        monitor.stop_monitoring()
        logging.debug(json.dumps(monitor.gpu_memory_logs))
        logging.debug(json.dumps(monitor.cpu_memory_logs))
        logging.debug(json.dumps(monitor.ram_used_mb))
        max_mem, start_mem, end_mem = monitor.log_gpu_metrics()
        cpu_max, cpu_start, cpu_end = monitor.log_cpu_metrics()
        metrics["memory_metrics"] = {
            "peak_gpu_mem": max_mem - start_mem,
            "peak_cpu_mem": cpu_max - cpu_start,
        }

    return metrics


def get_search_metrics(workloadToExecute, indexType, indexingParam, xq, gt):
    parameters_level_metrics = []
    graph_file = get_graph_file(workloadToExecute, indexType, indexingParam)

    for searchParam in workloadToExecute["search-parameters"]:
        logging.info(
            f"=== Running search for index config: {indexingParam} and search config: {searchParam}==="
        )
        searchTimingMetrics = search_indices.runIndicesSearch(
            xq, graph_file, searchParam, gt
        )
        logging.info(f"===== Timing Metrics : {searchTimingMetrics} ====")
        logging.info(
            f"=== Completed search for index config: {indexingParam} and search config: {searchParam}==="
        )
        logging.info("=======")
        parameters_level_metrics.append(
            {
                "indexing-params": indexingParam,
                "search-timing-metrics": searchTimingMetrics,
                "search-params": searchParam,
            }
        )
        logging.info("Sleeping for 5 sec for better metrics capturing")
        time.sleep(5)

    return parameters_level_metrics
