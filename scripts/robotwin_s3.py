from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_HEAD_NOT_FOUND_RE = re.compile(r"(?:\bNoSuchKey\b|\bNot Found\b|(?<!\d)404(?!\d))", re.IGNORECASE)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return the bucket and normalized prefix from an s3:// URI."""
    parsed = urlsplit(uri)
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"S3 URI must not contain a query or fragment: {uri!r}")

    prefix = parsed.path.strip("/")
    return parsed.netloc, prefix


@dataclass(frozen=True)
class S3Config:
    root_uri: str
    profile: str
    region: str
    credentials_file: str
    aws_cli: str = "aws"


class AwsCliS3:
    def __init__(self, config: S3Config) -> None:
        self.config = config
        self.bucket, self.prefix = parse_s3_uri(config.root_uri)
        self._env = os.environ.copy()
        self._env.update(
            {
                "AWS_SHARED_CREDENTIALS_FILE": str(Path(config.credentials_file).expanduser()),
                "AWS_PROFILE": config.profile,
                "AWS_REGION": config.region,
                "AWS_DEFAULT_REGION": config.region,
                "AWS_MAX_ATTEMPTS": "10",
                "AWS_RETRY_MODE": "adaptive",
            }
        )

    def key(self, relative: str) -> str:
        """Join a safe relative object key beneath the configured root prefix."""
        if not isinstance(relative, str):
            raise TypeError("relative key must be a string")
        if not relative or relative.startswith("/") or "\\" in relative:
            raise ValueError(f"Invalid relative S3 key: {relative!r}")

        parts = relative.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Invalid relative S3 key: {relative!r}")

        return "/".join(filter(None, (self.prefix, relative)))

    def uri(self, relative: str) -> str:
        return f"s3://{self.bucket}/{self.key(relative)}"

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            args,
            shell=False,
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    @staticmethod
    def _raise_for_error(result: subprocess.CompletedProcess[bytes]) -> None:
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )

    def head(self, relative: str) -> dict[str, Any] | None:
        result = self._run(
            [
                self.config.aws_cli,
                "s3api",
                "head-object",
                "--bucket",
                self.bucket,
                "--key",
                self.key(relative),
                "--output",
                "json",
            ]
        )
        if result.returncode != 0:
            error_text = result.stderr.decode("utf-8", errors="replace")
            if _HEAD_NOT_FOUND_RE.search(error_text):
                return None
            self._raise_for_error(result)

        try:
            value = json.loads(result.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Invalid JSON from aws s3api head-object for {self.uri(relative)}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Unexpected head-object response for {self.uri(relative)}")
        return value

    def read_bytes(self, relative: str) -> bytes:
        result = self._run(
            [
                self.config.aws_cli,
                "s3",
                "cp",
                self.uri(relative),
                "-",
                "--only-show-errors",
            ]
        )
        self._raise_for_error(result)
        return result.stdout

    def upload_file(
        self,
        local: str | os.PathLike[str],
        relative: str,
        sha256: str,
    ) -> dict[str, Any]:
        local_path = Path(local)
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        if not _SHA256_RE.fullmatch(sha256):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")

        expected_size = local_path.stat().st_size
        expected_sha256 = sha256.lower()
        digest = hashlib.sha256()
        with local_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Local payload changed before upload: {local_path}; "
                f"expected sha256={expected_sha256}, got {actual_sha256}"
            )
        if local_path.stat().st_size != expected_size:
            raise RuntimeError(f"Local payload size changed before upload: {local_path}")
        result = self._run(
            [
                self.config.aws_cli,
                "s3",
                "cp",
                str(local_path),
                self.uri(relative),
                "--only-show-errors",
                "--checksum-algorithm",
                "SHA256",
                "--metadata",
                f"sha256={expected_sha256}",
            ]
        )
        self._raise_for_error(result)

        uploaded = self.head(relative)
        if uploaded is None:
            raise RuntimeError(f"Uploaded object is missing: {self.uri(relative)}")

        content_length = uploaded.get("ContentLength")
        metadata = uploaded.get("Metadata")
        remote_sha256 = metadata.get("sha256") if isinstance(metadata, dict) else None
        if content_length != expected_size or remote_sha256 != expected_sha256:
            raise RuntimeError(
                f"Uploaded object verification failed for {self.uri(relative)}: "
                f"expected size={expected_size}, sha256={expected_sha256}; "
                f"got size={content_length!r}, sha256={remote_sha256!r}"
            )
        return uploaded

    def delete(self, relative: str) -> None:
        result = self._run(
            [
                self.config.aws_cli,
                "s3",
                "rm",
                self.uri(relative),
                "--only-show-errors",
            ]
        )
        self._raise_for_error(result)
