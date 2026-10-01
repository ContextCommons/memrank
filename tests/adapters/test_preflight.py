import httpx
import pytest

from memrank.adapters.preflight import PreflightError, preflight
from tests.fakes import FakeAdapter


def test_preflight_passes_for_reachable_adapter():
    preflight(FakeAdapter(name="ok"))  # no raise


def test_preflight_fails_loud_with_engine_and_url():
    class Broken(FakeAdapter):
        base_url = "http://localhost:9"

        def retrieve(self, *a, **k):
            raise httpx.ConnectError("refused")

    with pytest.raises(PreflightError, match="broken.*http://localhost:9"):
        preflight(Broken(name="broken"))


def test_preflight_never_quotes_a_credential_in_the_address():
    class Leaky(FakeAdapter):
        base_url = "http://operator:hunter2@localhost:9?api_key=zzz"

        def retrieve(self, *a, **k):
            raise RuntimeError("engine said no")

    with pytest.raises(PreflightError) as caught:
        preflight(Leaky(name="leaky"))

    message = str(caught.value)
    assert "localhost:9" in message
    for leaked in ("operator", "hunter2", "api_key", "zzz"):
        assert leaked not in message
