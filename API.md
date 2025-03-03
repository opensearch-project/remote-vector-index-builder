# Remote Vector Service - API Contract

## Endpoints

### Trigger Build

```
POST /_build

Request Parameters
{
    "repository_type": "s3", // Derived from repository settings
    "container_name": "VectorBucketName", // Derived from repository settings
    "vector_path": "VectorBlobPath", // File path all vector blobs are written to
    "doc_id_path" : "DocIdPath", // File path all doc Ids are written to
    "tenant_id": "UniqueClusterID", // Unique identifier for the cluster making the request
    "dimension": 768,
    "doc_count": 1000000,
    "data_type": "float",
    "engine": "faiss",
    "index_parameters": {
        "space_type": // {l2, innerproduct}
        "algorithm": "hnsw",
        "algorithm_parameters": {
            "ef_construction": 100,
            "ef_search": 100,
            "m": 16
        }
    }
}

Request Response:
{
    "job_id" : "String" // Unique identifier per build
}
```

#### Index Parameters

* The only required parameters are `repository_type`, `container_name`, `vector_path`, `doc_id_path`, `dimension`, and `doc_count`. The rest will follow k-NN's existing precedent for defaults.
* The top two parameters are derived by the KNN plugin from the customer's repo setting. `container_name` is used to specifically refer to the "bucket" (or non-S3 equivalent), rather than the name of the repository itself.
* Tenant ID is included to be used for billing, authorization, etc
* Dimension, doc count, and data type are specifically placed on the first JSON level so the build service can quickly use them first to calculate workload size
* Engine: If in the future we have different workers hosting different engines, like Lucene etc. this parameter will act as an extension point. This is not a required parameter.
* Space type is inside `index_parameters` since it is only used in index creation
* By including an `algorithm` field in index parameters, we leave the door open for IVF, future algorithms
* The existing algorithm parameters will be type checked on the client and server side, but the size of the map is variable to allow for future parameter additions
* Qualitative parameters like `repository_type`, `data_type`, `engine`, `algorithm`, and `space_type` currently only support the options listed. The other numerical and string settings follow k-NN/repository snapshot precedent on ranges and expected values: https://opensearch.org/docs/latest/search-plugins/knn/knn-index/

#### Error codes

- `500 Internal Server Error` if there’s an unexpected issue while processing the request. 
- `507 Out of Memory Exception` if worker does not have memory to handle the request. 
- `409 Request Conflict` if a conflicting request exists.

### Get Status

```
GET /_status/{job_id}

Request Response:
{
    "task_status" : "String", //RUNNING_INDEX_BUILD, FAILED_INDEX_BUILD, COMPLETED_INDEX_BUILD
    "index_path" : "String" // Null if not completed
    "error_message": "String"
}
```

Client can expect an error in “error_message” if task_status == `FAILED_INDEX_BUILD`.


#### Error codes
- `404 Not Found` if job ID does not exist. 


### Authentication

If authentication is configured, the endpoint must support Basic Authentication. The server is expected to return a 401 Unauthorized status code if authentication is enabled and credentials are missing/incorrect. If there is no authentication configured, the server will process requests with no Authentication header.

* * *

## Error Handling

#### Retriable Request Errors

Some HTTP requests may receive a response with an error status code. The client will agree to resend an HTTP request that receives any of the following errors:

| Status Code | Description | Notes |
|-------------|-------------|--------|
| 408 | Request Timeout | The server took too long to respond. Retrying may help |
| 429 | Too Many Requests | Indicates rate-limiting. Retry after the delay specified in the Retry-After header |
| 500 | Internal Server Error | A server-side issue that may resolve itself. Retrying is optional |
| 502 | Bad Gateway | Suggests a temporary network issue or service stack disruption that may self-correct |
| 503 | Service Unavailable | May be due to temporary service outages or in-progress deployments |
| 504 | Gateway Timeout | A downstream server (e.g., DNS) didn’t respond in time. Retrying may resolve the issue |
| 509 | Bandwidth Limit Exceeded | Server has exceeded its allocated bandwidth limit |


#### Non-Retriable Request Errors

The following failure codes have been designated to signify that the build request should be abandoned. These status codes will throw an exception with a descriptive error, and the node will fall back to CPU build. Most will require intervention (bad auth, bad endpoint, unhealthy build service)
| Status Code | Description | Notes |
|-------------|-------------|--------|
| 400 | Bad Request | The client sent an invalid request. Fix the issue in the request before trying again. |
| 401 | Unauthorized | Authentication is required or failed. Fix the authentication issue before trying again (e.g. get a fresh token). |
| 403 | Forbidden | The client lacks permission to access the resource. Retrying will not change the server’s response. Fix the authorization issue such as getting a new a token with additional scopes before trying again. |
| 404 | Not Found | The requested resource does not exist. Retrying will not succeed unless the resource becomes available later due to some background processing. |
| 405 | Method Not Allowed | The HTTP method used is not supported. Retrying with the same method will not resolve the issue. |
| 409 | Conflict | Indicates a conflict in the request such as unique constraints for referential integrity. Retrying without addressing the conflict will continue to fail. |
| 422 | Unprocessable Entity | The server understands the request but cannot process it due to semantic errors. Fix the issue in the request before trying again. |

_Error code descriptions source: https://www.restapitutorial.com/advanced/responses/retries_