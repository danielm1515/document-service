"""Where accepted PDFs go (HospitalAgent design §4.4): a private S3 bucket, or memory in tests."""
from __future__ import annotations

from typing import Any, Protocol


class StorageFailed(Exception):
    """The file could not be stored. The reason is an error code, never the content."""


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...


class InMemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data


class S3ObjectStore:
    """Private, encrypted at rest (SSE-S3), over TLS (boto3's default endpoint is https). The IAM
    user behind the credentials may only PutObject/GetObject under patients/* (design §4.4)."""

    def __init__(self, bucket: str, region: str, *, client: Any = None) -> None:
        self.bucket, self.region = bucket, region
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("s3", region_name=region,
                                  config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}))
        self._client = client

    def __repr__(self) -> str:
        return f"S3ObjectStore(bucket={self.bucket!r}, region={self.region!r})"  # never credentials

    def put(self, key: str, data: bytes) -> None:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data,
                                    ContentType="application/pdf", ServerSideEncryption="AES256")
        except ClientError as exc:
            raise StorageFailed(exc.response.get("Error", {}).get("Code", "ClientError")) from None
        except BotoCoreError as exc:
            raise StorageFailed(type(exc).__name__) from None
