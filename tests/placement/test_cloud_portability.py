"""A cloud task can only resolve the targets shipped with memrank.

`targets.path` and the operator's own directory exist on ONE machine. A ref that resolves here can
be unresolvable in a cloud task, and it costs a Fargate task to find out: on 2026-08-13
`mem0:memory-arena` -- defined in a sibling repo on the submitter's laptop -- launched and its
container exited 1 on its first line with `unknown target 'mem0:memory-arena'`. Nothing was
misconfigured, and no newer harness image would have helped.

The property under test is that the refusal happens BEFORE anything is submitted, at every door.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from memrank.targets.portability import local_only, unportable_message

LOCAL_VARIANT = """\
schema_version: 1
name: mem0:on-my-laptop
from: mem0
components: {embedder: {provider: openai, model: text-embedding-3-large, dims: 1536}}
"""
LOCAL_BASE = """\
schema_version: 1
name: houseblend
abstract: true
kind: stack
engine: {artifact: hindsight, port: 8888}
compose: hindsight.compose.yaml
service: engine
interface: {adapter: hindsight, transport: http}
components: {llm: {provider: anthropic, model: claude-sonnet-4-5-20250929}}
"""


def path_dir(tmp_path, monkeypatch, files: dict[str, str]) -> Path:
    """A directory of descriptors on targets.path, isolated from the operator's real config."""
    directory = tmp_path / "elsewhere" / "targets"
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (directory / f"{name}.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("MEMRANK_TARGETS_PATH", str(directory))
    return directory


def test_a_shipped_target_is_portable(tmp_path, monkeypatch):
    path_dir(tmp_path, monkeypatch, {"mem0-local": LOCAL_VARIANT})

    assert local_only(["hindsight:matched", "mem0", "word-overlap"]) == {}


def test_a_target_defined_on_this_machine_is_not(tmp_path, monkeypatch):
    """Today's case, exactly: a variant of a shipped base, living in a sibling repo."""
    directory = path_dir(tmp_path, monkeypatch, {"mem0-local": LOCAL_VARIANT})

    assert local_only(["mem0:on-my-laptop"]) == {"mem0:on-my-laptop": directory}


def test_a_shipped_target_inheriting_a_local_base_is_not(tmp_path, monkeypatch):
    """The ref itself looks shipped. Only the chain walk sees where its engine came from."""
    directory = path_dir(tmp_path, monkeypatch,
                         {"base": LOCAL_BASE,
                          "child": "schema_version: 1\nname: brew\nfrom: houseblend\n"})

    assert local_only(["brew"]) == {"brew": directory}


def test_a_local_file_shadowing_a_shipped_name_is_not(tmp_path, monkeypatch):
    """`load_all` is documented "later sources win", so a local `hindsight` replaces the shipped one
    and the ref stays spelled exactly as before. The submitter would ship a name the cloud resolves
    to DIFFERENT bytes -- worse than the failure this module exists to prevent, because it succeeds.
    """
    directory = path_dir(tmp_path, monkeypatch, {
        "hindsight": LOCAL_BASE.replace("name: houseblend", "name: hindsight")
                               .replace("abstract: true\n", "")})

    assert local_only(["hindsight"]) == {"hindsight": directory}


def test_an_unknown_ref_is_left_to_the_resolver(tmp_path, monkeypatch):
    """It is not unportable, it does not exist -- and the resolver's error names what IS known."""
    path_dir(tmp_path, monkeypatch, {})

    assert local_only(["no-such-target"]) == {}


def test_the_message_names_the_directory_and_the_way_out():
    """One way out, not two.

    It used to also advise moving the manifest into `memrank/targets/builtin/`. Some targets are
    outside the package ON PURPOSE -- `mem0:voyage` and `mem0:bge-tei` were moved there deliberately
    on 2026-08-19 -- so that advice told an author to undo a decision rather than describing a limit.
    """
    message = unportable_message({"mem0:on-my-laptop": Path("/repos/evals/targets")})

    assert "mem0:on-my-laptop" in message and "/repos/evals/targets" in message
    assert "--on local" in message
    assert "memrank/targets/builtin/" not in message


def test_the_message_agrees_with_itself_about_number():
    one = unportable_message({"a": Path("/x")})
    two = unportable_message({"a": Path("/x"), "b": Path("/y")})

    assert "target is" in one and "targets are" in two


#: Every module that builds a cloud submission payload, found by the key the API requires of one.
#: An enumeration rather than a list, because per-door defence is leaky by construction: the point
#: is that a THIRD cloud door fails this test on the day it is added, not that today's two pass.
def _submission_modules() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "memrank"
    found = [path for path in root.rglob("*.py")
             if '"config": remote_config(' in path.read_text(encoding="utf-8")]
    assert found, "no cloud submission site found -- has the payload key been renamed?"
    return found


@pytest.mark.parametrize("module", _submission_modules(),
                         ids=lambda p: os.path.join(p.parent.name, p.name))
def test_every_cloud_submission_site_consults_the_guard(module):
    """The chokepoint rule, enforced rather than documented.

    `runner._submit_sweep` (the CLI) and `application.submission._submit_cloud` (the browser) reach
    the cloud by different code, and a guard applied in one is a guard bypassed by the other.
    """
    source = module.read_text(encoding="utf-8")

    assert "local_only" in source, (
        f"{module.name} builds a cloud submission payload but never calls local_only(); a target "
        f"defined only on the submitting machine would launch a task that cannot resolve it")
