## Scripts

This folder contains helpful scripts for testing purposes. Some common use cases are given below:

### Test memory usage, performance, and recall end-to-end

It can be helpful to test the impact of your remote vector index builder 
code changes on memory, performance, and recall with real datasets. Here is how you can do this:

1. Provision an EC2 instance with GPUs, following the [developer guide](https://github.com/opensearch-project/remote-vector-index-builder/blob/main/DEVELOPER_GUIDE.md)
2. Create a conda environment to run the scripts. Then activate the conda environment
   ```
   conda create -n my_env -c conda-forge -c pytorch -c nvidia -c rapidsai python=3.12 faiss-gpu-cuvs=1.12.0 py3nvml pandas matplotlib psutil numpy tqdm pyyaml h5py boto3
   ```
   ```
   conda activate my_env
   ```
3. Create a s3 bucket
4. Use the `dataset_to_s3` script to convert a hdf5 dataset to binary, and upload the dataset
to the s3 bucket. Example:
    ```
    python dataset_to_s3.py  --download-url "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/ms_marco-384-1m.hdf5?download=true"  --dataset-name "ms-marco-384"  --bucket "<your-bucket-name>"   --vectors-key "<path-to-vectors>.knnvec"  --docids-key "<path-to-doc-ids>.knndid"   --region "us-west-2"
    ```
   - You can use any of the datasets provided in the [benchmarks.yml](https://github.com/opensearch-project/remote-vector-index-builder/blob/main/benchmarking/benchmarks.yml)
   - This will also download the hdf5 dataset to the `/benchmarking/files/default_run_id/dataset` folder 
5. Start the memory profiling script. It will run in the background, until you press enter:
    ```
    python monitor_memory.py --identifier <filename-identifier> --monitor-gpu
    ```
   After you press 'enter', the script will output the peak GPU and CPU memory, and write
   the CPU and GPU memory snapshots (with timestamps) to a csv file
6. Build and run the remote vector index builder API docker image,
   following the [developer guide](https://github.com/opensearch-project/remote-vector-index-builder/blob/main/DEVELOPER_GUIDE.md)
   - When running the docker image with `docker run`, set `-e LOG_LEVEL=DEBUG` env variable to view the debug logs. 
     - These logs will tell you how long each step of the remote build (download from s3, index build, upload to s3) took
   - By default, the faiss index is stored on disk before it is uploaded to s3. To serialize the
   faiss index in memory instead of writing to disk, you can set `-e INDEX_SERIALIZATION_MODE=memory` env variable
7. Trigger a build request for the API image. Example:
   ```
   curl -XPOST "http://0.0.0.0:80/_build" \
   -H 'Content-Type: application/json' \
   -d '
   {
   "repository_type": "s3",
   "container_name": "<your-bucket-name>",
   "vector_path": "<path-to-vectors>.knnvec",
   "doc_id_path": "<path-to-doc-ids>.knndid",
   "dimension": "384",
   "doc_count": "1000000"
   }
   '
   ```
8. Once the build is complete, stop the `monitor_memory` script by pressing enter. You can now view the CPU and GPU memory usage stats
9. Use the `s3_recall_test` script to test the recall of your uploaded faiss index in s3. Example:
   ```
   python scripts/s3_recall_test.py --bucket <your-bucket-name> --index-s3-key <path-to-graph>.faiss --query-dataset /benchmarking/files/default_run_id/dataset/ms-marco-384.hdf5 --ef-search 256 --k 10 --region "us-west-2"
   ```


