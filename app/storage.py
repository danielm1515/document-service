"""Where accepted PDFs go (HospitalAgent design §4.4): a private S3 bucket, or memory in tests."""
from __future__ import annotations

from typing import Any, Protocol


class StorageFailed(Exception):
    """The file could not be stored. The reason is an error code, never the content."""


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> None: ...


class InMemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}

    def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        self.objects[key] = data
        self.content_types[key] = content_type


class S3ObjectStore:
    """Private, encrypted at rest (SSE-S3), over TLS (boto3's default endpoint is https). The IAM
    user behind the credentials may only PutObject/GetObject under patients/* (design §4.4)."""

    def __init__(self, bucket: str, region: str, *, client: Any = None) -> None:
        self.bucket, self.region = bucket, region
        if client is None:
            import boto3
            from botocore.config import Config
            # total_max_attempts counts the first try too: at most 2 sends, so an upload's S3
            # part is bounded by 2 x (5 s connect + 15 s read) = 40 s (README, "זמן תגובה מרבי").
            # (botocore's max_attempts would count retries only - 2 of them, 3 sends in all.)
            client = boto3.client("s3", region_name=region,
                                  config=Config(connect_timeout=5, read_timeout=15,
                                                retries={"total_max_attempts": 2, "mode": "standard"}))
        self._client = client

    def __repr__(self) -> str:
        return f"S3ObjectStore(bucket={self.bucket!r}, region={self.region!r})"  # never credentials

    def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data,
                                    ContentType=content_type, ServerSideEncryption="AES256")
        except ClientError as exc:
            raise StorageFailed(exc.response.get("Error", {}).get("Code", "ClientError")) from None
        except BotoCoreError as exc:
            raise StorageFailed(type(exc).__name__) from None
