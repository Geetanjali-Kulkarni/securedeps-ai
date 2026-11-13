import pytest
import requests

from securedeps.llm import AnthropicLLM, GeminiLLM, LLMError, get_llm


class Resp:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self._body, self.ok, self.bad_json = status, body, status < 400, bad_json

    def json(self):
        if self.bad_json:
            raise ValueError("no json")
        return self._body


class Net:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


GEM_OK = {"candidates": [{"content": {"parts": [{"text": '{"a":'}, {"text": '1}'}]}}]}


def test_gemini_request_shape_and_parsing():
    net = Net(Resp(body=GEM_OK))
    out = GeminiLLM("SECRETKEY", "gemini-test", net).complete("SYS", "USER")
    assert out == '{"a":1}'
    url, headers, body = net.calls[0]
    assert url.endswith("/models/gemini-test:generateContent")
    assert "SECRETKEY" not in url and headers["x-goog-api-key"] == "SECRETKEY"   # key never in the URL
    assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert body["contents"][0]["parts"][0]["text"] == "USER"
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_anthropic_request_shape_and_parsing():
    net = Net(Resp(body={"content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}))
    out = AnthropicLLM("KEY", None, net).complete("SYS", "USER")
    assert out == "hello world"
    url, headers, body = net.calls[0]
    assert url == "https://api.anthropic.com/v1/messages" and headers["x-api-key"] == "KEY"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-5-5" and body["system"] == "SYS"
    assert body["messages"] == [{"role": "user", "content": "USER"}]


@pytest.mark.parametrize("status,fragment", [(429, "rate limit"), (401, "rejected"), (403, "rejected"),
                                             (404, "model name"), (500, "HTTP 500")])
def test_http_errors_become_readable(status, fragment):
    for cls in (GeminiLLM, AnthropicLLM):
        with pytest.raises(LLMError, match=fragment):
            cls("k", "m", Net(Resp(status))).complete("s", "u")


def test_network_bad_json_and_blocked_answers():
    with pytest.raises(LLMError, match="network"):
        GeminiLLM("k", "m", Net(requests.ConnectionError())).complete("s", "u")
    with pytest.raises(LLMError, match="not JSON"):
        GeminiLLM("k", "m", Net(Resp(bad_json=True))).complete("s", "u")
    with pytest.raises(LLMError, match="SAFETY"):
        GeminiLLM("k", "m", Net(Resp(body={"promptFeedback": {"blockReason": "SAFETY"}}))).complete("s", "u")


def test_provider_selection():
    assert get_llm({}) is None
    assert isinstance(get_llm({"GEMINI_API_KEY": "g"}), GeminiLLM)
    assert isinstance(get_llm({"ANTHROPIC_API_KEY": "a"}), AnthropicLLM)
    both = {"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}
    assert isinstance(get_llm(both), GeminiLLM)
    assert isinstance(get_llm({**both, "LLM_PROVIDER": "anthropic"}), AnthropicLLM)
    assert get_llm({"GEMINI_API_KEY": "g", "LLM_PROVIDER": "anthropic"}) is None   # asked for a missing key


def test_timeout_is_retried_then_reported_readably():
    import requests
    calls = []

    class Slow:
        def post(self, *a, **k):
            calls.append(1)
            raise requests.Timeout()
    with pytest.raises(LLMError, match="too long"):
        GeminiLLM("K", "m", Slow()).complete("S", "U")
    assert len(calls) == 3


def test_timeout_then_success():
    import requests
    state = {"n": 0}

    class Flaky:
        def post(self, *a, **k):
            state["n"] += 1
            if state["n"] == 1:
                raise requests.Timeout()
            return Resp(body=GEM_OK)
    assert GeminiLLM("K", "m", Flaky()).complete("S", "U") == '{"a":1}'


def test_503_is_retried_then_succeeds():
    seq = [Resp(status=503), Resp(body=GEM_OK)]

    class Overloaded:
        def post(self, *a, **k):
            return seq.pop(0)
    assert GeminiLLM("K", "m", Overloaded()).complete("S", "U") == '{"a":1}'


def test_persistent_503_gives_readable_error():
    class Down:
        def post(self, *a, **k):
            return Resp(status=503)
    with pytest.raises(LLMError, match="HTTP 503"):
        GeminiLLM("K", "m", Down()).complete("S", "U")


def test_429_waits_and_retries_then_succeeds():
    seq = [Resp(status=429), Resp(body=GEM_OK)]

    class Limited:
        def post(self, *a, **k):
            return seq.pop(0)
    assert GeminiLLM("K", "m", Limited()).complete("S", "U") == '{"a":1}'


def test_persistent_429_gives_readable_error():
    class Limited:
        def post(self, *a, **k):
            return Resp(status=429)
    with pytest.raises(LLMError, match="rate limit"):
        GeminiLLM("K", "m", Limited()).complete("S", "U")


def test_http_400_error_shows_the_providers_message_and_key_hint():
    body = {"error": {"message": "API key not valid. Please pass a valid API key."}}
    with pytest.raises(LLMError) as e:
        GeminiLLM("k", session=Net(Resp(400, body))).complete("s", "u")
    assert "API key not valid" in str(e.value) and "export GEMINI_API_KEY" in str(e.value)
