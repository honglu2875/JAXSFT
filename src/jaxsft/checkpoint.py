"""Bounded checkpoint I/O primitives used by experimental sharded loaders."""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping


_CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")


@dataclass(frozen=True)
class HTTPRangeRecord:
    start: int
    end: int
    bytes_read: int
    elapsed_seconds: float
    total_size_bytes: int


class StrictHTTPRangeReader:
    """Read only exact HTTP byte intervals and fail if a server ignores Range.

    The response is rejected before its body is read unless it is HTTP 206 with
    an exact ``Content-Range``.  A one-byte over-read guard catches malformed
    or decompressed responses without accidentally consuming a full shard.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 60.0,
        maximum_request_bytes: int = 64 * 1024 * 1024,
        headers: Mapping[str, str] | None = None,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not url.startswith("https://"):
            raise ValueError("checkpoint range URL must use https")
        if timeout_seconds <= 0 or maximum_request_bytes <= 0:
            raise ValueError("range timeout and maximum request size must be positive")
        supplied_headers = dict(headers or {})
        reserved = {name.lower() for name in supplied_headers} & {"range", "accept-encoding"}
        if reserved:
            raise ValueError(f"caller may not override reserved range headers: {sorted(reserved)}")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.maximum_request_bytes = maximum_request_bytes
        self._headers = supplied_headers
        self._opener = opener
        self._records: list[HTTPRangeRecord] = []
        self._total_size_bytes: int | None = None

    @property
    def records(self) -> tuple[HTTPRangeRecord, ...]:
        return tuple(self._records)

    @property
    def bytes_read(self) -> int:
        return sum(record.bytes_read for record in self._records)

    @property
    def total_size_bytes(self) -> int | None:
        return self._total_size_bytes

    def read(self, start: int, end: int) -> bytes:
        """Read the inclusive interval ``[start, end]``."""

        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError(f"invalid inclusive HTTP range: {(start, end)}")
        expected_bytes = end - start + 1
        if expected_bytes > self.maximum_request_bytes:
            raise ValueError(
                f"HTTP range asks for {expected_bytes} bytes, above limit {self.maximum_request_bytes}"
            )
        request = urllib.request.Request(
            self.url,
            headers={
                **self._headers,
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
                "User-Agent": "JAXSFT-bounded-range-loader/0.1",
            },
            method="GET",
        )
        started = time.monotonic()
        with self._opener(request, timeout=self.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if status != 206:
                raise ValueError(f"range server returned HTTP {status}, expected 206; body was not read")
            content_range = response.headers.get("Content-Range")
            match = _CONTENT_RANGE.fullmatch(content_range or "")
            if match is None:
                raise ValueError(f"invalid Content-Range header: {content_range!r}")
            actual_start, actual_end, total_size = (int(value) for value in match.groups())
            if (actual_start, actual_end) != (start, end):
                raise ValueError(
                    f"Content-Range interval {(actual_start, actual_end)} does not match {(start, end)}"
                )
            if total_size <= end:
                raise ValueError(f"invalid Content-Range total size {total_size} for end offset {end}")
            if self._total_size_bytes is not None and total_size != self._total_size_bytes:
                raise ValueError(
                    f"source size changed from {self._total_size_bytes} to {total_size} between reads"
                )
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise ValueError(
                    f"Content-Length {content_length} does not match requested {expected_bytes} bytes"
                )
            payload = response.read(expected_bytes + 1)
        if len(payload) != expected_bytes:
            raise ValueError(f"range response has {len(payload)} bytes, expected {expected_bytes}")
        self._total_size_bytes = total_size
        self._records.append(
            HTTPRangeRecord(
                start=start,
                end=end,
                bytes_read=len(payload),
                elapsed_seconds=time.monotonic() - started,
                total_size_bytes=total_size,
            )
        )
        return payload


__all__ = ["HTTPRangeRecord", "StrictHTTPRangeReader"]
