import pytest

from jaxsft.checkpoint import StrictHTTPRangeReader


class _Response:
    def __init__(self, *, status, headers, payload):
        self.status = status
        self.headers = headers
        self.payload = payload
        self.read_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size):
        self.read_calls.append(size)
        return self.payload[:size]


def test_strict_http_range_reader_accepts_only_exact_partial_content():
    responses = []

    def opener(request, *, timeout):
        assert timeout == 7
        assert request.get_header("Range") == "bytes=10-13"
        assert request.get_header("Accept-encoding") == "identity"
        response = _Response(
            status=206,
            headers={"Content-Range": "bytes 10-13/100", "Content-Length": "4"},
            payload=b"abcd",
        )
        responses.append(response)
        return response

    reader = StrictHTTPRangeReader(
        "https://example.test/model.safetensors",
        timeout_seconds=7,
        maximum_request_bytes=8,
        opener=opener,
    )
    assert reader.read(10, 13) == b"abcd"
    assert responses[0].read_calls == [5]
    assert reader.bytes_read == 4
    assert reader.total_size_bytes == 100
    assert reader.records[0].start == 10
    assert reader.records[0].end == 13


def test_strict_http_range_reader_rejects_ignored_or_malformed_ranges_without_full_read():
    ignored = _Response(status=200, headers={"Content-Length": "100"}, payload=b"x" * 100)
    reader = StrictHTTPRangeReader(
        "https://example.test/model.safetensors",
        opener=lambda *_args, **_kwargs: ignored,
    )
    with pytest.raises(ValueError, match="body was not read"):
        reader.read(0, 3)
    assert ignored.read_calls == []

    malformed = _Response(
        status=206,
        headers={"Content-Range": "bytes 1-4/100", "Content-Length": "4"},
        payload=b"abcd",
    )
    reader = StrictHTTPRangeReader(
        "https://example.test/model.safetensors",
        opener=lambda *_args, **_kwargs: malformed,
    )
    with pytest.raises(ValueError, match="does not match"):
        reader.read(0, 3)
    assert malformed.read_calls == []


def test_strict_http_range_reader_enforces_scheme_headers_and_size_bound():
    with pytest.raises(ValueError, match="https"):
        StrictHTTPRangeReader("http://example.test/model")
    with pytest.raises(ValueError, match="reserved"):
        StrictHTTPRangeReader("https://example.test/model", headers={"Range": "bytes=0-1"})
    reader = StrictHTTPRangeReader("https://example.test/model", maximum_request_bytes=3)
    with pytest.raises(ValueError, match="above limit"):
        reader.read(0, 3)
