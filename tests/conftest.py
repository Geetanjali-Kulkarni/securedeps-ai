

import pytest


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retries wait between attempts; tests must not."""
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Tests must not depend on the developer's shell (API keys, debug switches, model names)."""
    for name in ("SECUREDEPS_DEBUG", "GEMINI_API_KEY", "GEMINI_MODEL", "ANTHROPIC_API_KEY",
                 "ANTHROPIC_MODEL", "LLM_PROVIDER", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
