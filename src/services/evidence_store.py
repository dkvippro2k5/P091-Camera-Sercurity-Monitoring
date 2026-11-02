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
            # No public base URL configured, so the object is not reachable from a browser
            # directly. Hand back the same authenticated API path the local backend uses and
            # let `fetch` below stream it — an `s3://` URL here would reach the frontend as an
            # `<img src>` no browser can load.
            return f"/api/v1/evidence/{evidence_id}"

        destination = self.local_dir / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(jpeg)
        os.replace(temporary, destination)
        return f"/api/v1/evidence/{evidence_id}"

    def local_path(self, evidence_id: str) -> Path:
        return self.local_dir / "evidence" / f"{evidence_id}.jpg"

    def exists(self, evidence_id: str) -> bool:
        """Whether a stored frame is there, without transferring it.

        The evidence frames of a past alert are not recorded in any table -- they are written
        under the derived key `{alert_id}-f{n}` and described only on the in-memory Alert,
        which is gone once the ring buffer rolls. Reconstructing the strip for an older
        incident therefore means probing those keys, and probing with `fetch` would download
        every frame just to learn it exists. `head_object` asks the same question for none of
        the bytes.
        """
        if self.backend == "s3":
            try:
                self._s3.head_object(Bucket=self.bucket, Key=f"evidence/{evidence_id}.jpg")
                return True
            except Exception:  # noqa: BLE001 - a missing key and a transport error both mean "cannot show it"
                return False
        return self.local_path(evidence_id).is_file()

    def fetch(self, evidence_id: str) -> bytes | None:
        """Read one stored evidence JPEG back, or None when it is not there.

        Exists so the API can serve S3-backed evidence through its own authenticated route
        rather than requiring a public bucket. Without it, a deployment with
        `EVIDENCE_STORAGE_BACKEND=s3` and no `EVIDENCE_S3_PUBLIC_BASE_URL` has no way to show
        an operator the frame at all.
        """
        if self.backend == "s3":
            try:
                response = self._s3.get_object(Bucket=self.bucket, Key=f"evidence/{evidence_id}.jpg")
                return response["Body"].read()
            except Exception as exc:  # noqa: BLE001 - a missing key and a transport error both mean "cannot show it"
                logger.warning(f"Could not fetch evidence {evidence_id} from S3: {exc}")
                return None

        path = self.local_path(evidence_id)
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            logger.warning(f"Could not read evidence {evidence_id}: {exc}")
            return None
