import csv
import json
import logging
import os
import sys
from benchmarking.data_types.data_types import IndexTypes, WorkloadTypes
from benchmarking.utils.common_utils import (
    ensureDir,
    formatTimingMetricsValue,
    readAllWorkloads,
)


def persistMetricsAsCSV(
    workloadType: WorkloadTypes,
    allMetrics: dict,
    workloadName: str,
    indexType: IndexTypes,
):
    file_path = ensureDir(f"results/{workloadName}")
    fields = [
        "workload-name",
        "indexType",
        "dataset-name",
        "dimensions",
        "vectors-count",
        "queries-count",
        "indexing-params",
        "index-creation-time",
        "gpu-to-cpu-index-conversion-time",
        "write-to-file-time",
        "write-index-time",
        "total-build-time",
        "peak-gpu-memory-usage",
        "peak-cpu-memory-usage",
        "search-parameter",
        "search-time",
        "unit",
        "search-throughput",
        "recall@100",
        "recall@1",
    ]
    rows = []
    if workloadType == WorkloadTypes.INDEX:
        logging.error("This type of workload is not supported for writing data in csv")
        sys.exit()
    else:
        workloadDetails = allMetrics[workloadName]["workload-details"]
        searchParamItr = 0
        indexingParamItr = 0
        for searchMetric in allMetrics[workloadName]["searchMetrics"]:
            searchParamItr = searchParamItr + 1
            searchTimingMetrics = searchMetric["search-timing-metrics"]
            row = {
                "workload-name": workloadName,
                "indexType": indexType.value,
                "dataset-name": workloadDetails["dataset_name"],
                "dimensions": workloadDetails["dimension"],
                "queries-count": workloadDetails.get("queriesCount"),
                "vectors-count": workloadDetails.get("vectorsCount"),
                "indexing-params": searchMetric["indexing-params"],
                "search-time": formatTimingMetricsValue(
                    searchTimingMetrics["searchTime"]
                ),
                "unit": searchTimingMetrics["units"],
                "recall@100": searchTimingMetrics["recall_at_100"],
                "recall@1": searchTimingMetrics["recall_at_1"],
                "search-parameter": searchMetric["search-params"],
                "search-throughput": searchTimingMetrics["search_throughput"],
            }

            if allMetrics[workloadName].get("indexingMetrics") is not None:
                row["vectors-count"] = workloadDetails["vectorsCount"]
                indexingMetrics = allMetrics[workloadName]["indexingMetrics"]
                row["index-creation-time"] = formatTimingMetricsValue(
                    indexingMetrics[indexingParamItr]["indexing-timingMetrics"][
                        "indexTime"
                    ]
                )
                row["write-index-time"] = formatTimingMetricsValue(
                    indexingMetrics[indexingParamItr]["indexing-timingMetrics"][
                        "writeIndexTime"
                    ]
                )
                row["gpu-to-cpu-index-conversion-time"] = formatTimingMetricsValue(
                    indexingMetrics[indexingParamItr]["indexing-timingMetrics"].get(
                        "gpu_to_cpu_index_conversion_time"
                    )
                )
                row["write-to-file-time"] = formatTimingMetricsValue(
                    indexingMetrics[indexingParamItr]["indexing-timingMetrics"].get(
                        "write_to_file_time"
                    )
                )
                row["total-build-time"] = formatTimingMetricsValue(
                    indexingMetrics[indexingParamItr]["indexing-timingMetrics"][
                        "totalTime"
                    ]
                )
                row["peak-gpu-memory-usage"] = formatTimingMetricsValue(
                    indexingMetrics[indexingParamItr]["memory_metrics"]["peak_gpu_mem"]
                )
                row["peak-cpu-memory-usage"] = formatTimingMetricsValue(
                    indexingMetrics[indexingParamItr]["memory_metrics"]["peak_cpu_mem"]
                )
                if searchParamItr % len(workloadDetails["search-parameters"]) == 0:
                    indexingParamItr = indexingParamItr + 1
            rows.append(row)

    with open(
        f"{file_path}/{workloadType.value}_{indexType.value}.csv", "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        # writing headers (field names)
        writer.writeheader()
        # writing data rows
        writer.writerows(rows)

    logging.info(
        f"Results are stored at location: {file_path}/{workloadType.value}_{indexType.value}.csv"
    )
    return f"{file_path}/{workloadType.value}_{indexType.value}.csv"


def writeDataInCSV(workloadNames: str, indexType: str, workloadType: WorkloadTypes):
    if workloadType == WorkloadTypes.INDEX:
        logging.error("This type of workload is not supported for writing data in csv")
        sys.exit()

    indexTypesList = []

    if indexType == "all":
        indexTypesList = IndexTypes.enumList()
    else:
        indexTypesList.append(IndexTypes.from_str(indexType))

    workloadCSVFiles = []
    for indexTypeEnum in indexTypesList:
        if len(workloadNames) == 0:
            allWorkloads = readAllWorkloads()

            for currentWorkloadName in allWorkloads[indexTypeEnum.value]:
                csvFile = writeDataInCSVPerWorkload(
                    currentWorkloadName, indexTypeEnum, workloadType
                )
                if csvFile is not None:
                    workloadCSVFiles.append(csvFile)
        else:
            for workloadName in workloadNames:
                csvFile = writeDataInCSVPerWorkload(
                    workloadName, indexTypeEnum, workloadType
                )
                if csvFile is not None:
                    workloadCSVFiles.append(csvFile)

    writeDataInSingleCSVFile(workloadCSVFiles, "all_results.csv")


def writeDataInCSVPerWorkload(
    workloadName: str, indexType: IndexTypes, workloadType: WorkloadTypes
) -> str:
    dir = ensureDir(f"results/{workloadName}")
    jsonFile = f"{dir}/{workloadType.value}_{indexType.value}.json"
    if os.path.exists(jsonFile) is False:
        logging.warn(
            f"No result file exist for {workloadName} , "
            f"indexType: {indexType.value} workloadType {workloadType.value} at {jsonFile}"
        )
        return None

    f = open(jsonFile)
    allMetrics = json.load(f)
    f.close
    return persistMetricsAsCSV(workloadType, allMetrics, workloadName, indexType)


def writeDataInSingleCSVFile(workloadCSVFiles: list, outfileName: str):
    if len(workloadCSVFiles) == 0:
        logging.warn("No CSV files to combine to a single result file")
        return
    dir = ensureDir("results/all/")

    if os.path.exists(f"{dir}/{outfileName}"):
        logging.info(f"Deleting the file results/all/{outfileName}, as it exist")
        os.remove(f"{dir}/{outfileName}")

    outputFile = open(f"{dir}/{outfileName}", "w")
    # This will add header and other all the data from first file in output file.
    with open(workloadCSVFiles[0]) as f:
        logging.info(f"Writing file: {workloadCSVFiles[0]}")
        for line in f:
            outputFile.write(line)

    # Now we call add all other files by skipping their headers
    for resultFiles in workloadCSVFiles[1:]:
        logging.info(f"Writing file: {resultFiles}")
        with open(resultFiles) as f:
            next(f)
            for line in f:
                outputFile.write(line)
    logging.info(f"All data is written in the file results/all/{outfileName}")
