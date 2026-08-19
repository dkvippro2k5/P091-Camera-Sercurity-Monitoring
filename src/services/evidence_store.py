"""Durable incident evidence storage with local and S3-compatible backends."""

import io
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class EvidenceStore:
    def __init__(self, backend: str, local_dir: str, bucket: str = "", endpoint_url: str = "", region: str = "", public_base_url: str = ""):
        self.backend = backend
        self.local_dir = Path(local_dir)
        self.bucket = bucket
        self.public_base_url = public_base_url.rstrip("/")
        self._s3 = None
        if backend == "s3":
            import boto3

            self._s3 = boto3.client("s3", endpoint_url=endpoint_url or None, region_name=region or None)
        elif backend != "local":
            raise ValueError(f"Unsupported evidence storage backend: {backend}")

    def put(self, evidence_id: str, jpeg: bytes) -> str:
        key = f"evidence/{evidence_id}.jpg"
        if self.backend == "s3":
            self._s3.upload_fileobj(io.BytesIO(jpeg), self.bucket, key, ExtraArgs={"ContentType": "image/jpeg"})
            if self.public_base_url:
                return f"{self.public_base_url}/{key}"
            return f"s3://{self.bucket}/{key}"

        destination = self.local_dir / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(jpeg)
        os.replace(temporary, destination)
        return f"/api/v1/evidence/{evidence_id}"

    def local_path(self, evidence_id: str) -> Path:
        return self.local_dir / "evidence" / f"{evidence_id}.jpg"
