import boto3
import pytest
from botocore.stub import Stubber

from app.storage import InMemoryObjectStore, S3ObjectStore, StorageFailed


def test_in_memory_store_keeps_what_it_is_given():
    store = InMemoryObjectStore()
    store.put("patients/P-1/DOC-1.pdf", b"%PDF-1")
    assert store.objects == {"patients/P-1/DOC-1.pdf": b"%PDF-1"}


def s3_client():
    return boto3.client("s3", region_name="eu-north-1", aws_access_key_id="x", aws_secret_access_key="y")


def test_s3_puts_a_private_encrypted_pdf():
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_response("put_object", {}, {"Bucket": "b", "Key": "patients/P-1/DOC-1.pdf", "Body": b"%PDF",
                                              "ContentType": "application/pdf", "ServerSideEncryption": "AES256"})
        S3ObjectStore("b", "eu-north-1", client=client).put("patients/P-1/DOC-1.pdf", b"%PDF")
        stub.assert_no_pending_responses()


def test_an_s3_failure_is_storage_failed():
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_client_error("put_object", service_error_code="AccessDenied", http_status_code=403)
        with pytest.raises(StorageFailed):
            S3ObjectStore("b", "eu-north-1", client=client).put("k", b"%PDF")


def test_the_store_never_shows_credentials():
    assert repr(S3ObjectStore("bucket-name", "eu-north-1", client=s3_client())) == \
        "S3ObjectStore(bucket='bucket-name', region='eu-north-1')"
