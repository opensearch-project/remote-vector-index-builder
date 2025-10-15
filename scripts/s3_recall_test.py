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
import boto3
import tempfile

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.search.search_indices import runIndicesSearch
from benchmarking.dataset.dataset_utils import prepare_search_dataset

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_index_from_s3(bucket: str, key: str, local_path: str, region: str = 'us-east-1'):
    """Download index file from S3 to local path"""
    s3_client = boto3.client('s3', region_name=region)
    s3_client.download_file(bucket, key, local_path)
    logging.info(f"Downloaded index from s3://{bucket}/{key} to {local_path}")

def main():
    parser = argparse.ArgumentParser(description='Download index from S3 and test recall')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--index-s3-key', required=True, help='S3 key for the index file')
    parser.add_argument('--query-dataset', required=True, help='Path to query dataset (HDF5)')
    parser.add_argument('--ef-search', type=int, default=100, help='ef_search parameter')
    parser.add_argument('--k', type=int, default=100, help='K value for recall')
    parser.add_argument('--normalize', action='store_true', help='Normalize query vectors')
    parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()
    setup_logging()

    try:
        # Create temporary file for index
        with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as temp_file:
            temp_index_path = temp_file.name

        # Download index from S3
        download_index_from_s3(args.bucket, args.index_s3_key, temp_index_path, args.region)

        # Prepare search dataset
        d, query_vectors, ground_truth = prepare_search_dataset(args.query_dataset, args.normalize)

        # Run search and calculate recall
        search_params = {
            'ef_search': args.ef_search,
            'K': args.k
        }

        results = runIndicesSearch(query_vectors, temp_index_path, search_params, ground_truth)

        # Print results
        logging.info("Search Results:")
        for key, value in results.items():
            logging.info(f"{key}: {value}")

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary file
        if 'temp_index_path' in locals() and os.path.exists(temp_index_path):
            os.unlink(temp_index_path)
            logging.info(f"Cleaned up temporary index file: {temp_index_path}")

if __name__ == "__main__":
    main()

