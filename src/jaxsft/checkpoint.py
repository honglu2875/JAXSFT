"""Bounded checkpoint I/O primitives used by experimental sharded loaders."""

from __future__ import annotations

import re
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import requests


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


class _RequestsResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    url: str
    raw: Any

    def close(self) -> None: ...


class _RequestsSession(Protocol):
    def get(self, url: str, **kwargs: Any) -> _RequestsResponse: ...

    def close(self) -> None: ...


class _ResolvedURLExpired(Exception):
    pass


class StrictPooledHTTPRangeReader:
    """Thread-safe strict range reader with bounded persistent connections.

    Hugging Face shard URLs redirect to immutable, signed CDN URLs.  Calling
    :meth:`resolve` follows that redirect exactly once; later reads go directly
    to the resolved URL and avoid spending one Hub resolver request per tensor.
    A fixed session pool bounds both concurrency and live HTTPS connections.

    As with :class:`StrictHTTPRangeReader`, response headers are validated
    before the body is consumed, and a one-byte over-read guard rejects a
    malformed or decoded response.  Signed resolved URLs are intentionally not
    exposed through records or ``repr``-friendly public state.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 60.0,
        maximum_request_bytes: int = 768 * 1024 * 1024,
        connections: int = 16,
        maximum_attempts: int = 4,
        headers: Mapping[str, str] | None = None,
        session_factory: Callable[[], _RequestsSession] = requests.Session,
    ) -> None:
        if not url.startswith("https://"):
            raise ValueError("checkpoint range URL must use https")
        if timeout_seconds <= 0 or maximum_request_bytes <= 0:
            raise ValueError("range timeout and maximum request size must be positive")
        if isinstance(connections, bool) or not isinstance(connections, int) or connections <= 0:
            raise ValueError("range connection count must be a positive integer")
        if (
            isinstance(maximum_attempts, bool)
            or not isinstance(maximum_attempts, int)
            or maximum_attempts <= 0
        ):
            raise ValueError("range attempt count must be a positive integer")
        supplied_headers = dict(headers or {})
        reserved = {name.lower() for name in supplied_headers} & {"range", "accept-encoding"}
        if reserved:
            raise ValueError(f"caller may not override reserved range headers: {sorted(reserved)}")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.maximum_request_bytes = maximum_request_bytes
        self.connections = connections
        self.maximum_attempts = maximum_attempts
        self._headers = supplied_headers
        self._sessions: queue.LifoQueue[_RequestsSession] = queue.LifoQueue(connections)
        for _ in range(connections):
            session = session_factory()
            if isinstance(session, requests.Session):
                retry = requests.adapters.Retry(
                    total=maximum_attempts - 1,
                    connect=maximum_attempts - 1,
                    read=maximum_attempts - 1,
                    status=maximum_attempts - 1,
                    other=maximum_attempts - 1,
                    allowed_methods=frozenset({"GET"}),
                    status_forcelist=(429, 500, 502, 503, 504),
                    backoff_factor=0.5,
                    respect_retry_after_header=True,
                    raise_on_status=False,
                )
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=1,
                    pool_maxsize=1,
                    max_retries=retry,
                    pool_block=True,
                )
                session.mount("https://", adapter)
            self._sessions.put(session)
        self._records: list[HTTPRangeRecord] = []
        self._total_size_bytes: int | None = None
        self._resolved_url: str | None = None
        self._state_lock = threading.Lock()
        self._resolve_lock = threading.Lock()
        self._closed = False

    @property
    def records(self) -> tuple[HTTPRangeRecord, ...]:
        with self._state_lock:
            return tuple(self._records)

    @property
    def bytes_read(self) -> int:
        with self._state_lock:
            return sum(record.bytes_read for record in self._records)

    @property
    def total_size_bytes(self) -> int | None:
        with self._state_lock:
            return self._total_size_bytes

    @property
    def resolved(self) -> bool:
        with self._state_lock:
            return self._resolved_url is not None

    def _validate_range(self, start: int, end: int) -> int:
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
        return expected_bytes

    def _read_from(
        self,
        url: str,
        start: int,
        end: int,
        *,
        allow_redirects: bool,
    ) -> tuple[bytes, str]:
        expected_bytes = self._validate_range(start, end)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("range reader is closed")
        request_headers = {
            **self._headers,
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "JAXSFT-pooled-range-loader/0.1",
        }
        session = self._sessions.get()
        response: _RequestsResponse | None = None
        started = time.monotonic()
        try:
            response = session.get(
                url,
                headers=request_headers,
                timeout=self.timeout_seconds,
                stream=True,
                allow_redirects=allow_redirects,
            )
            status = response.status_code
            if not allow_redirects and status in {401, 403}:
                raise _ResolvedURLExpired
            if status != 206:
                raise ValueError(
                    f"range server returned HTTP {status}, expected 206; body was not read"
                )
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
            with self._state_lock:
                if self._total_size_bytes is not None and total_size != self._total_size_bytes:
                    raise ValueError(
                        f"source size changed from {self._total_size_bytes} "
                        f"to {total_size} between reads"
                    )
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise ValueError(
                    f"Content-Length {content_length} does not match requested {expected_bytes} bytes"
                )
            response.raw.decode_content = False
            payload = response.raw.read(expected_bytes + 1)
            final_url = response.url
        finally:
            if response is not None:
                response.close()
            self._sessions.put(session)
        if len(payload) != expected_bytes:
            raise ValueError(f"range response has {len(payload)} bytes, expected {expected_bytes}")
        if not final_url.startswith("https://"):
            raise ValueError("resolved checkpoint range URL must use https")
        elapsed = time.monotonic() - started
        with self._state_lock:
            self._total_size_bytes = total_size
            self._records.append(
                HTTPRangeRecord(
                    start=start,
                    end=end,
                    bytes_read=len(payload),
                    elapsed_seconds=elapsed,
                    total_size_bytes=total_size,
                )
            )
        return payload, final_url

    def _resolve_locked(self) -> None:
        with self._state_lock:
            if self._resolved_url is not None:
                return
        _, final_url = self._read_from(self.url, 0, 0, allow_redirects=True)
        with self._state_lock:
            self._resolved_url = final_url

    def resolve(self) -> None:
        """Resolve and cache the signed CDN URL using a one-byte range read."""

        with self._resolve_lock:
            self._resolve_locked()

    def read(self, start: int, end: int) -> bytes:
        """Read inclusive interval ``[start, end]`` from the cached CDN URL."""

        for attempt in range(2):
            self.resolve()
            with self._state_lock:
                resolved_url = self._resolved_url
            assert resolved_url is not None
            try:
                payload, final_url = self._read_from(
                    resolved_url,
                    start,
                    end,
                    allow_redirects=False,
                )
            except _ResolvedURLExpired:
                if attempt:
                    raise ValueError("resolved checkpoint URL expired again after refresh") from None
                with self._resolve_lock:
                    with self._state_lock:
                        if self._resolved_url == resolved_url:
                            self._resolved_url = None
                    self._resolve_locked()
                continue
            if final_url != resolved_url:
                raise ValueError("resolved checkpoint URL unexpectedly redirected")
            return payload
        raise AssertionError("unreachable range URL refresh path")

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        sessions = [self._sessions.get() for _ in range(self.connections)]
        for session in sessions:
            session.close()

    def __enter__(self) -> StrictPooledHTTPRangeReader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["HTTPRangeRecord", "StrictHTTPRangeReader", "StrictPooledHTTPRangeReader"]
