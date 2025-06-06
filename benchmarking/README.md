### Benchmarking

This folder contains the code needed to benchmark Faiss GPU vs CPU build, using a docker container
on a GPU machine. This code does not use `opensearch-benchmark`, it uses a custom `main.py`
to run the benchmarks.

### How to run the code

1. Provision an instance
    - See [instance provisioning instructions](https://github.com/opensearch-project/remote-vector-index-builder/blob/main/DEVELOPER_GUIDE.md#provisioning-an-instance-for-development)
   on how to do this
    - Note that if you are only benchmarking CPU Faiss code with an EC2 instance, you can use a non-GPU instance like r5.2xlarge, 
   and a non-GPU AMI like AL2023
2. Clone the repository to the instance
    ```angular2html
    git clone https://github.com/opensearch-project/remote-vector-index-builder.git
    ```
3. Change into the benchmarking directory:
    ```angular2html
    cd benchmarking
    ```
4. Build the docker image. 
Note that any image tag can be used, not just `opensearchstaging/remote-vector-index-builder:benchmark-tool-latest`
Also, you probably will need to run all of the `docker` commands with `sudo`. Simply do `sudo <docker command>` 
    ```
    docker build -t opensearchstaging/remote-vector-index-builder:benchmark-tool-latest .
    ```
5. Create a docker mountpoint directory, for the benchmarking results. 
    ```
    mkdir docker-mountpoint
    ```
6. Create a file to hold environment variables. Call this file 'env_variables'
    ```
    touch env_variables
    ```
    - The environment variables are:
       - `workload`: name of the dataset. Can be a comma separated list of datasets, 
       to benchmark with multiple datasets. Please see the [`benchmarks.yml`](benchmarks.yml) file for the list of 
       possible datasets. The name of the dataset is the top level yaml key - for example, `sift-128`, `ms-marco-384`, etc
       - `workload_type`: Can be `INDEX`, `INDEX_AND_SEARCH`, or `INDEX`. Defaults to `INDEX_AND_SEARCH`
       - `index_type`: Can be `gpu`, `cpu`, or `all`. Defaults to `all`
       - `run_id`: Sub-folder to store results. Defaults to `default_run_id`
       - `run_type`: Can be `run_workload`, `write_results` or `all`. Defaults to `all`. 
          - `run_workload` will run the benchmarks and generate all graphs, and save the results in json files.
          `write_results` will read the json files, and combine them into a single csv file. 
       - For example, to run the GPU Faiss benchmarks with `sift-128` and `gist-960` for indexing and searching, 
       the environment variables file will look like:
           ```
           index_type=gpu
           workload=sift-128,gist-960
           ```
7. Run the docker container: `docker run --env-file env_variables -v ./docker-mountpoint:/benchmarking/files --gpus all opensearchstaging/remote-vector-index-builder:benchmark-tool-latest`
    - You can run the docker container in the background with `-d` option
    - Note that downloading the sift-128 and gist-960 datasets may fail. In that case, manually download 
    the files to the `./docker_mountpoint/{run_id}/dataset` folder using `wget <download_url>`
8. All files will be stored in `./docker_mountpoint/{run_id}` directory. Results will be available in `./docker_mountpoint/{run_id}/results`. Logs are available at `./docker_mountpoint/vector_search.log`

### Tips

- If you run into `permissions denied` errors when running any of the shell commands, the easiest thing to do is to become root user by doing `sudo bash`. Then try re-running the command. 
- To test different benchmarking configurations, edit the `indexing-parameters` section of the [`benchmarks.yml`](benchmarks.yml) file. Then re-build the docker image, and re-run the image. 
  - Any of the [`ivf_pq_params`](https://github.com/opensearch-project/remote-vector-index-builder/blob/4454886e7f74a32c4d105355f4239dc0b0616b67/remote_vector_index_builder/core/common/models/index_builder/faiss/ivf_pq_build_cagra_config.py#L14)
  or [`faiss_cagra_config`](https://github.com/opensearch-project/remote-vector-index-builder/blob/4454886e7f74a32c4d105355f4239dc0b0616b67/remote_vector_index_builder/core/common/models/index_builder/faiss/faiss_gpu_index_cagra_builder.py#L28)
  hyperparameters can be tested
  - The default configuration will take a lot of time to finish, since it tests many different hyperparameter combinations.
- You can change the logging configuration to `logging.debug` in [`main.py`](main.py) file, to log the GPU and memory usage snapshots in `./docker_mountpoint/vector_search.log`