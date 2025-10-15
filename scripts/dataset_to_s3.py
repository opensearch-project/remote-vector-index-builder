# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
import boto3
from io import BytesIO

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.dataset.dataset_utils import downloadDataSet, prepare_indexing_dataset

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_binary(data) -> bytes:
    """Convert numpy array or list to binary format"""
    if isinstance(data, list):
        data = np.array(data, dtype=np.int32)
    return data.tobytes()

def upload_to_s3(data: bytes, bucket: str, key: str, region: str = 'us-east-1'):
    """Upload binary data to S3"""
    s3_client = boto3.client('s3', region_name=region)
    buffer = BytesIO(data)
    s3_client.upload_fileobj(buffer, bucket, key)
    logging.info(f"Uploaded {len(data)} bytes to s3://{bucket}/{key}")

def main():
    parser = argparse.ArgumentParser(description='Download HDF5 dataset, convert vectors and doc IDs to binary, and upload to S3')
    parser.add_argument('--download-url', required=True, help='URL to download the dataset')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--vectors-key', required=True, help='S3 key for the vectors binary file')
    parser.add_argument('--docids-key', required=True, help='S3 key for the doc IDs binary file')
    parser.add_argument('--compressed', action='store_true', help='Dataset is compressed')
    parser.add_argument('--compression-type', default='bz2', help='Compression type (default: bz2)')
    parser.add_argument('--normalize', action='store_true', help='Normalize vectors')
    parser.add_argument('--doc-count', type=int, default=-1, help='Number of documents to read (-1 for all)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()
    setup_logging()

    try:
        # Download dataset
        dataset_path = downloadDataSet(
            args.download_url,
            args.dataset_name,
            args.compressed,
            args.compression_type if args.compressed else None
        )

        # Prepare indexing dataset
        d, vectors, ids = prepare_indexing_dataset(dataset_path, args.normalize, args.doc_count)

        # Convert to binary
        vectors_binary = convert_to_binary(vectors)
        docids_binary = convert_to_binary(ids)

        # Upload vectors to S3
        upload_to_s3(vectors_binary, args.bucket, args.vectors_key, args.region)

        # Upload doc IDs to S3
        upload_to_s3(docids_binary, args.bucket, args.docids_key, args.region)

        logging.info(f"Successfully processed {len(vectors)} vectors of dimension {d}")
        logging.info(f"Vectors uploaded to: s3://{args.bucket}/{args.vectors_key}")
        logging.info(f"Doc IDs uploaded to: s3://{args.bucket}/{args.docids_key}")

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

