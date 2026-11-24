from types import ModuleType, SimpleNamespace
import sys

import pytest

from vllm_ascend import spec_decode

class NativeProposer:
    def __init__(self, config, device, runner):
        self.runner = runner

class ExternalProposer(NativeProposer):
    pass

@pytest.fixture
def selector(monkeypatch):
    monkeypatch.setattr(spec_decode, "AscendEagleProposer", NativeProposer)
    config = SimpleNamespace(speculative_config=SimpleNamespace(use_step3p5_mtp=lambda: False))
    runner = object()
    return lambda method: spec_decode.get_spec_decode_method(method, config, "cpu", runner)

def test_default_proposer(monkeypatch, selector):
    monkeypatch.delenv("VLLM_ASCEND_EXTERNAL_MTP_PROPOSER", raising=False)
    assert type(selector("mtp")) is NativeProposer

@pytest.mark.parametrize("method", ["eagle", "eagle3"])
def test_other_methods_keep_their_proposer(monkeypatch, selector, method):
    monkeypatch.setenv("VLLM_ASCEND_EXTERNAL_MTP_PROPOSER", "absent_draft_provider:make")
    assert type(selector(method)) is NativeProposer

def test_external_mtp_factory(monkeypatch, selector):
    mod = ModuleType("glm53_test_draft_provider")
    mod.make = ExternalProposer
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setenv("VLLM_ASCEND_EXTERNAL_MTP_PROPOSER", mod.__name__ + ":make")
    assert type(selector("mtp")) is ExternalProposer
    mod.make = lambda *args: object()
    with pytest.raises(TypeError):
        selector("mtp")
    monkeypatch.delattr(mod, "make")
    with pytest.raises(ValueError):
        selector("mtp")
