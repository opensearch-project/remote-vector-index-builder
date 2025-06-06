import os
import shutil
from urllib.request import urlretrieve

from benchmarking.decorators.timer import timer_func
from benchmarking.dataset.dataset import HDF5DataSet, Context
import numpy as np
import logging
import bz2
import sys

from benchmarking.utils.common_utils import ensureDir


def downloadDataSetForWorkload(workloadToExecute: dict) -> str:
    download_url = workloadToExecute["download_url"]
    dataset_name = workloadToExecute["dataset_name"]
    isCompressed = False if workloadToExecute.get("compressed") is None else True
    compressionType = workloadToExecute.get("compression-type")
    return downloadDataSet(download_url, dataset_name, isCompressed, compressionType)


def downloadDataSet(
    download_url: str,
    dataset_name: str,
    isCompressed: bool,
    compressionType: str | None,
) -> str:
    logging.info("Downloading dataset...")
    destination_path_compressed = None
    dir_path = ensureDir("dataset")
    if compressionType is not None:
        destination_path_compressed = os.path.join(
            dir_path, f"{dataset_name}.hdf5.{compressionType}"
        )
    destination_path = os.path.join(dir_path, f"{dataset_name}.hdf5")

    if not os.path.exists(destination_path):
        if isCompressed:
            logging.info(
                f"downloading {download_url} -> {destination_path_compressed} ..."
            )
            urlretrieve(download_url, destination_path_compressed)
            decompress_dataset(
                destination_path_compressed, compressionType, destination_path
            )
        else:
            logging.info(f"downloading {download_url} -> {destination_path} ...")
            urlretrieve(download_url, destination_path)
        logging.info(f"downloaded {download_url} -> {destination_path}...")
    return destination_path


@timer_func
def decompress_dataset(filePath: str, compressionType: str, outputFile: str):
    logging.info(f"Decompression {filePath} having compression type: {compressionType}")
    if compressionType == "bz2":
        with bz2.BZ2File(filePath) as fr, open(outputFile, "wb") as fw:
            shutil.copyfileobj(fr, fw, length=1024 * 1024 * 10)  # read by 100MB chunks
        logging.info("Completed decompression... ")
    else:
        logging.error(
            f"Compression type : {compressionType} is not supported for decompression"
        )
        sys.exit()


@timer_func
def prepare_indexing_dataset(
    datasetFile: str, normalize: bool = None, docToRead: int = -1
) -> tuple[int, np.ndarray, list]:
    logging.info(f"Reading data set from file: {datasetFile}")
    index_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.INDEX)

    logging.info(
        f"Total number of docs that we will read for indexing: {index_dataset.size() if docToRead == -1 else docToRead}"
    )
    xb: np.ndarray = index_dataset.read(
        index_dataset.size() if docToRead == -1 or docToRead is None else docToRead
    ).astype(dtype=np.float32)
    d: int = len(xb[0])
    ids = [i for i in range(len(xb))]
    if normalize:
        logging.info("Doing normalization...")
        xb = xb / np.linalg.norm(xb)
        logging.info("Completed normalization...")

    logging.info("Dataset info : ")
    logging.info(f"Dimensions: {d}")
    logging.info(f"Total Vectors: {len(xb)}")
    logging.info(f"Total Ids: {len(ids)}")
    logging.info(f"Normalized: {normalize}")

    return d, xb, ids


@timer_func
def prepare_search_dataset(
    datasetFile: str, normalize: bool = None
) -> tuple[int, np.ndarray, HDF5DataSet]:
    logging.info(f"Reading data set from file: {datasetFile}")
    search_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.QUERY)
    xq: np.ndarray = search_dataset.read(search_dataset.size()).astype(dtype=np.float32)
    gt: HDF5DataSet = HDF5DataSet(datasetFile, Context.NEIGHBORS)
    d: int = len(xq[0])
    logging.info("Dataset info : ")
    logging.info(f"Dimensions: {d}")
    logging.info(f"Total Vectors: {len(xq)}")
    logging.info(f"Normalized: {normalize}")
    if normalize:
        logging.info("Doing normalization...")
        xq = xq / np.linalg.norm(xq)
        logging.info("Completed normalization...")
    return d, xq, gt
