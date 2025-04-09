# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from core.common.exceptions import BlobError
from core.object_store.s3.s3_object_store import S3ObjectStore, get_boto3_client


# Mock the logger to prevent actual logging during tests
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("core.object_store.s3.s3_object_store.logger"):
        yield


@pytest.fixture
def object_store_config():
    return {
        "retries": 3,
        "region": "us-west-2",
        "debug": False,
        "download_transfer_config": {"max_concurrency": 4},
        "upload_transfer_config": {"max_concurrency": 8},
    }


@pytest.fixture
def s3_object_store(index_build_parameters, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        yield store


@pytest.fixture
def bytes_buffer():
    bytes_buffer = BytesIO()
    yield bytes_buffer
    bytes_buffer.close()


def test_get_boto3_client():
    with patch("boto3.client") as mock_client:
        # Test caching behavior
        client1 = get_boto3_client("us-west-2", 3)
        client2 = get_boto3_client("us-west-2", 3)
        assert client1 == client2
        mock_client.assert_called_once()

        # Test different parameters create new client
        get_boto3_client("us-east-1", 3)
        assert mock_client.call_count == 2


def test_get_boto3_client_with_local_stack():
    with patch("boto3.client") as mock_client:
        # Test caching behavior
        get_boto3_client("us-west-2", 3, "t")
        args = mock_client.call_args.kwargs
        assert args["endpoint_url"] == S3ObjectStore.LOCAL_STACK_ENDPOINT

        get_boto3_client("us-west-2", 3, "f")
        args = mock_client.call_args.kwargs
        assert "endpoint_url" not in args

        get_boto3_client("us-west-2", 3, "false")
        args = mock_client.call_args.kwargs
        assert "endpoint_url" not in args

        get_boto3_client("us-west-2", 3, "0")
        args = mock_client.call_args.kwargs
        assert "endpoint_url" not in args


def test_s3_object_store_initialization(index_build_parameters, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        assert store.bucket == index_build_parameters.container_name
        assert store.max_retries == object_store_config["retries"]
        assert store.region == object_store_config["region"]
        assert (
            store.download_transfer_config["max_concurrency"]
            == object_store_config["download_transfer_config"]["max_concurrency"]
        )
        assert (
            store.upload_transfer_config["max_concurrency"]
            == object_store_config["upload_transfer_config"]["max_concurrency"]
        )
        assert not store.debug


# also test if os.cpu_count is none
def test_s3_object_store_initialization_debug_config(index_build_parameters):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        with patch("os.cpu_count", return_value=None):
            store = S3ObjectStore(index_build_parameters, {"debug": True})
            assert store.debug


def test_create_custom_config(index_build_parameters):
    custom_config = {
        "retries": 3,
        "region": "us-west-2",
        "debug": False,
        "download_transfer_config": {
            "multipart_chunksize": 20 * 1024 * 1024,
            "max_concurrency": 8,
            "param": "value",
        },
        "upload_transfer_config": {
            "max_concurrency": 8,
        },
        "download_args": {"ChecksumMode": "DISABLED", "param": "value"},
        "upload_args": {"ChecksumAlgorithm": "SHA1", "param": "value"},
    }

    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, custom_config)
        assert store.max_retries == 3
        assert store.region == "us-west-2"
        assert not store.debug
        assert store.download_transfer_config["multipart_chunksize"] == 20 * 1024 * 1024
        assert store.download_transfer_config["max_concurrency"] == 8
        assert store.download_args["ChecksumMode"] == "DISABLED"

        assert store.upload_transfer_config["max_concurrency"] == 8
        assert (
            store.upload_transfer_config["multipart_chunksize"]
            == store.DEFAULT_UPLOAD_TRANSFER_CONFIG["multipart_chunksize"]
        )
        assert store.upload_args["ChecksumAlgorithm"] == "SHA1"

        # In production, if boto3 does not support 'param', it will throw an exception
        # So, no need for the object store client to also validate the params, during construction
        assert "param" in store.download_transfer_config
        assert "param" in store.download_args
        assert "param" in store.upload_args


def test_read_blob_success(index_build_parameters, object_store_config, bytes_buffer):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        store.s3_client.download_fileobj = Mock()

        store.read_blob("test/path", bytes_buffer)
        store.s3_client.download_fileobj.assert_called_once()
        assert isinstance(
            store.s3_client.download_fileobj.call_args.kwargs["Config"], TransferConfig
        )
        # validate a transfer config parameter matches the object_store_config parameter
        assert (
            store.s3_client.download_fileobj.call_args.kwargs["Config"].__dict__[
                "max_concurrency"
            ]
            == store.download_transfer_config["max_concurrency"]
        )
        assert store.s3_client.download_fileobj.call_args.kwargs["Callback"] is None
        assert (
            store.s3_client.download_fileobj.call_args.kwargs["ExtraArgs"]
            == store.download_args
        )
        assert store.s3_client.download_fileobj.call_args.args == (
            store.bucket,
            "test/path",
            bytes_buffer,
        )


def test_read_blob_with_debug(
    index_build_parameters, object_store_config, bytes_buffer
):
    object_store_config["debug"] = True
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        store.s3_client.download_fileobj = Mock()

        store.read_blob("test/path", bytes_buffer)

        # Verify callback was passed
        callback = store.s3_client.download_fileobj.call_args.kwargs["Callback"]
        assert callback is not None
        # Test the callback directly
        assert store._read_progress == 0
        callback(100)  # Simulate 100 bytes transferred
        assert store._read_progress == 100
        callback(50)  # Simulate 50 more bytes
        assert store._read_progress == 150


def test_read_blob_client_error_failure(
    index_build_parameters, object_store_config, bytes_buffer
):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        error = ClientError(
            {"Error": {"Code": "LimitExceededException", "Message": "Limit Exceeded"}},
            "DownloadFileObj",
        )
        store.s3_client.download_fileobj.side_effect = error
        with pytest.raises(BlobError):
            store.read_blob("test/path", bytes_buffer)


def test_read_blob_type_error_failure(
    index_build_parameters, object_store_config, bytes_buffer
):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        error = TypeError(
            "TransferConfig.__init__() got an unexpected keyword argument"
        )
        store.s3_client.upload_file.side_effect = error
        store.s3_client.download_fileobj.side_effect = error
        with pytest.raises(BlobError):
            store.read_blob("test/path", bytes_buffer)


def test_write_blob_success(index_build_parameters, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        store.s3_client.upload_file = Mock()
        store.write_blob("local/path", "remote/path")

        store.s3_client.upload_file.assert_called_once()
        assert isinstance(
            store.s3_client.upload_file.call_args.kwargs["Config"], TransferConfig
        )
        # validate a transfer config parameter matches the object_store_config parameter
        assert (
            store.s3_client.upload_file.call_args.kwargs["Config"].__dict__[
                "max_concurrency"
            ]
            == store.upload_transfer_config["max_concurrency"]
        )
        assert store.s3_client.upload_file.call_args.kwargs["Callback"] is None
        assert (
            store.s3_client.upload_file.call_args.kwargs["ExtraArgs"]
            == store.upload_args
        )
        assert store.s3_client.upload_file.call_args.args == (
            "local/path",
            store.bucket,
            "remote/path",
        )


def test_write_blob_with_debug(index_build_parameters, object_store_config):
    object_store_config["debug"] = True
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        store.s3_client.upload_file = Mock()

        store.write_blob("local/path", "remote/path")

        # Verify callback was passed
        callback = store.s3_client.upload_file.call_args.kwargs["Callback"]
        assert callback is not None
        # Test the callback directly
        assert store._write_progress == 0
        callback(100)  # Simulate 100 bytes transferred
        assert store._write_progress == 100
        callback(50)  # Simulate 50 more bytes
        assert store._write_progress == 150


def test_write_blob_client_error_failure(index_build_parameters, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        error = ClientError(
            {"Error": {"Code": "LimitExceededException", "Message": "Limit Exceeded"}},
            "UploadFile",
        )
        store.s3_client.upload_file.side_effect = error
        with pytest.raises(BlobError):
            store.write_blob("local/path", "remote/path")


def test_write_blob_type_error_failure(index_build_parameters, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_parameters, object_store_config)
        error = TypeError(
            "TransferConfig.__init__() got an unexpected keyword argument"
        )
        store.s3_client.upload_file.side_effect = error
        with pytest.raises(BlobError):
            store.write_blob("local/path", "remote/path")
