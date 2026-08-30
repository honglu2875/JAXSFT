import pytest

from jaxsft.checkpoint import StrictHTTPRangeReader, StrictPooledHTTPRangeReader


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


class _Raw:
    def __init__(self, payload):
        self.payload = payload
        self.decode_content = True
        self.read_calls = []

    def read(self, size):
        self.read_calls.append(size)
        return self.payload[:size]


class _RequestsStyleResponse:
    def __init__(self, *, status, headers, payload, url):
        self.status_code = status
        self.headers = headers
        self.raw = _Raw(payload)
        self.url = url
        self.closed = False

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_pooled_range_reader_resolves_once_and_uses_direct_url():
    origin = "https://example.test/model.safetensors"
    resolved = "https://cdn.example.test/signed"
    calls = []
    all_responses = [
        _RequestsStyleResponse(
            status=206,
            headers={"Content-Range": "bytes 0-0/100", "Content-Length": "1"},
            payload=b"a",
            url=resolved,
        ),
        _RequestsStyleResponse(
            status=206,
            headers={"Content-Range": "bytes 10-13/100", "Content-Length": "4"},
            payload=b"bcde",
            url=resolved,
        ),
        _RequestsStyleResponse(
            status=206,
            headers={"Content-Range": "bytes 20-21/100", "Content-Length": "2"},
            payload=b"fg",
            url=resolved,
        ),
    ]
    responses = list(all_responses)
    sessions = []

    def factory():
        session = _Session(responses, calls)
        sessions.append(session)
        return session

    with StrictPooledHTTPRangeReader(
        origin,
        timeout_seconds=7,
        maximum_request_bytes=8,
        connections=1,
        session_factory=factory,
    ) as reader:
        reader.resolve()
        reader.resolve()
        assert reader.read(10, 13) == b"bcde"
        assert reader.read(20, 21) == b"fg"
        assert reader.resolved
        assert reader.total_size_bytes == 100
        assert reader.bytes_read == 7
    assert [call[0] for call in calls] == [origin, resolved, resolved]
    assert calls[0][1]["allow_redirects"] is True
    assert all(call[1]["headers"]["Accept-Encoding"] == "identity" for call in calls)
    assert calls[1][1]["headers"]["Range"] == "bytes=10-13"
    assert all(response.raw.decode_content is False for response in all_responses)
    assert all(response.closed for response in all_responses)
    assert sessions[0].closed


def test_pooled_range_reader_rejects_ignored_range_before_body_read():
    response = _RequestsStyleResponse(
        status=200,
        headers={"Content-Length": "100"},
        payload=b"x" * 100,
        url="https://example.test/model.safetensors",
    )
    session = _Session([response], [])
    reader = StrictPooledHTTPRangeReader(
        "https://example.test/model.safetensors",
        connections=1,
        session_factory=lambda: session,
    )
    with pytest.raises(ValueError, match="body was not read"):
        reader.resolve()
    assert response.raw.read_calls == []
    reader.close()


def test_pooled_range_reader_refreshes_an_expired_signed_url_once():
    origin = "https://example.test/model.safetensors"
    expired = "https://cdn.example.test/expired"
    renewed = "https://cdn.example.test/renewed"
    calls = []
    expired_response = _RequestsStyleResponse(
        status=403,
        headers={},
        payload=b"must not be read",
        url=expired,
    )
    responses = [
        _RequestsStyleResponse(
            status=206,
            headers={"Content-Range": "bytes 0-0/100", "Content-Length": "1"},
            payload=b"a",
            url=expired,
        ),
        expired_response,
        _RequestsStyleResponse(
            status=206,
            headers={"Content-Range": "bytes 0-0/100", "Content-Length": "1"},
            payload=b"a",
            url=renewed,
        ),
        _RequestsStyleResponse(
            status=206,
            headers={"Content-Range": "bytes 10-13/100", "Content-Length": "4"},
            payload=b"bcde",
            url=renewed,
        ),
    ]
    session = _Session(responses, calls)
    reader = StrictPooledHTTPRangeReader(
        origin,
        connections=1,
        session_factory=lambda: session,
    )
    assert reader.read(10, 13) == b"bcde"
    assert [call[0] for call in calls] == [origin, expired, origin, renewed]
    assert expired_response.raw.read_calls == []
    assert reader.bytes_read == 6
    reader.close()
