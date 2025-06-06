import os
from benchmarking.workload.workload import runWorkload
from benchmarking.data_types.data_types import WorkloadTypes
from benchmarking.results import writeDataInCSV
import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/benchmarking/files/vector_search.log"),
        logging.StreamHandler(),
    ],
)


def main():

    workload_names = os.environ.get("workload", [])
    index_type = os.environ.get("index_type", "all")
    workload_type = os.environ.get("workload_type", WorkloadTypes.INDEX_AND_SEARCH)
    run_id = os.environ.get("run_id", None)
    run_type = os.environ.get("run_type", "all")

    if len(workload_names) != 0:
        workload_names = workload_names.split(",")

    if run_id is not None:
        config.run_id = run_id

    if workload_type != WorkloadTypes.INDEX_AND_SEARCH:
        workload_type = WorkloadTypes.from_str(workload_type)

    logging.info(
        f"Running with workload: {workload_names}, "
        f"index_type: {index_type}, workload_type: {workload_type}, "
        f"run_id: {run_id}, run_type: {run_type}"
    )
    if run_type == "all" or run_type == "run_workload":
        runWorkload(workload_names, index_type, workload_type)
    if run_type == "all" or run_type == "write_results":
        writeDataInCSV(workload_names, index_type, workload_type)


if __name__ == "__main__":
    main()
