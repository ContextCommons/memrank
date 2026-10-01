# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""``memrank run``, ``memrank agents`` and ``memrank serve``: evaluating an agent.

    memrank run locomo --agent full-context --cases 5 --questions 4
    memrank run locomo --agent ./my-agent.yaml --concurrency 4
    memrank run locomo --agent ./my-agent.yaml --no-judge
    memrank run ./my-eval.yaml --agent ./my-agent.yaml
    memrank agents ls

``run`` needs a login (``memrank auth login``) and nothing started: it runs the evaluation in
its own process, starts the agent when the agent file says how, judges the answers with the
organisation's saved ``ANTHROPIC_API_KEY`` (``memrank secrets set ANTHROPIC_API_KEY --org
<org>``), and records the run in the organisation's run history -- registered as it starts, so
its link is printed first and its page follows it live (:mod:`memrank.loop.live`). ``serve`` exists only to share
one evaluation service between machines (``run --service URL``).
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import typer

from memrank import config, errors
from memrank.errors import ActionRequired, Concluded, MemrankError, optional_import
from memrank.judging.judge import JudgeConfig
from memrank.loop.judge import DEFAULT_JUDGE_MODEL
from memrank.loop.lanes import DEFAULT_CONCURRENCY
from memrank.outcome import Step
from memrank.term import style

DEFAULT_SERVICE_PORT = 8787
DEFAULT_AGENT_PORT = 8765
DEFAULT_OUTPUT = Path("results")
REFERENCE_AGENTS = ("full-context",)

agents_app = typer.Typer(help="The agents memrank ships, as baselines to measure yours against.",
                         no_args_is_help=True)


def run(
    ctx: typer.Context,
    benchmark: str | None = typer.Argument(None, metavar="EVALUATION",
                                           help="A shipped evaluation (`memrank evals ls`: "
                                                "locomo, longmemeval, beam:100k, ...) or the "
                                                "path to your evaluation file"),
    agent: str = typer.Option(..., "--agent", "-a",
                              help="A shipped agent's name (`memrank agents ls`) or the path "
                                   "to your agent file"),
    cases: int | None = typer.Option(None, "--cases", min=1, help="Sample this many cases"),
    questions: int | None = typer.Option(None, "--questions", min=1,
                                         help="Sample at most this many questions per case"),
    seed: int = typer.Option(0, "--seed", help="Seed for case and question sampling"),
    judge: bool = typer.Option(True, "--judge/--no-judge",
                               help="--no-judge records answers and reports NO quality score; "
                                    "judging otherwise uses your organisation's saved "
                                    "ANTHROPIC_API_KEY"),
    judge_model: str = typer.Option(DEFAULT_JUDGE_MODEL, "--judge-model",
                                    help="The Anthropic model that judges the answers"),
    concurrency: int = typer.Option(DEFAULT_CONCURRENCY, "--concurrency", "-j", min=1,
                                    help="Run this many cases at once; each case's steps "
                                         "stay in order. 1 runs them one at a time"),
    resume: str | None = typer.Option(None, "--resume",
                                      help="Continue a run by its id. A failed case or "
                                           "answer stays failed unless --retry-failed"),
    retry_failed: bool = typer.Option(False, "--retry-failed",
                                      help="With --resume: run every case with a failed "
                                           "reset, feed or answer again from a fresh reset "
                                           "under a new session id, asking only the questions "
                                           "still owed; successful answers are kept and not "
                                           "paid for again"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, "--output-dir",
                                    help="Each run writes one folder here"),
    service: str | None = typer.Option(None, "--service",
                                       help="Use a shared evaluation service at this URL "
                                            "instead of evaluating in this process"),
) -> None:
    """Evaluate an agent on a shipped evaluation or your own. Each run writes one folder."""
    from memrank.judging.client import JUDGE_MODEL_PREFIX
    from memrank.loop.ending import Invocation, ended
    from memrank.loop.report import done
    from memrank.loop.run import RunOptions
    from memrank.loop.trace import RunTrace
    from memrank.term import outcome

    if (benchmark is None) == (resume is None):
        raise typer.BadParameter("name an EVALUATION for a new run, or --resume a run "
                                 "(not both)")
    if retry_failed and resume is None:
        raise typer.BadParameter("--retry-failed retries the failed cases of a run you "
                                 "--resume; a new run has none")
    if not judge_model.startswith(JUDGE_MODEL_PREFIX):
        raise typer.BadParameter(f"the judge is an Anthropic model; --judge-model must start "
                                 f"with {JUDGE_MODEL_PREFIX!r}")
    options = RunOptions(evaluation=_evaluation_ref(benchmark), cases=cases, questions=questions,
                         seed=seed, resume=resume, retry_failed=retry_failed, judge=judge,
                         judge_model=judge_model, concurrency=concurrency)
    trace = RunTrace()
    call = Invocation(command=_command(ctx), resume=lambda run_id: _resume_command(ctx, run_id))
    try:
        finished = _evaluate(ctx, call, trace, agent, options, output_dir, service)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - every ending is shown
        if errors.wants_traceback() and not isinstance(exc, MemrankError | KeyboardInterrupt):
            raise
        ending, code = ended(exc, trace, call)
        raise Concluded(ending, code) from exc
    outcome.render(done(finished.result, trace, finished.url, finished.org))


def _evaluate(ctx: typer.Context, call, trace, agent: str, options, output_dir: Path,  # noqa: ANN001
              service: str | None):  # noqa: ANN202 - the loop's types, imported lazily
    """Read the agent, sign in, and run it; ``call`` learns the agent and org as they are."""
    from memrank.agents import resolve
    from memrank.connect.spec import load_agent
    from memrank.loop.run import Instruments, run_agent
    from memrank.loop.upload import connect

    # A shared service's resumed run has no evaluation here to read: its key is fetched.
    needs_model = True
    if options.evaluation:  # a broken evaluation file is refused here, before anything runs
        needs_model = _needs_model(options.evaluation, options.seed)
    spec = load_agent(resolve(agent))
    call.agent = spec
    evaluator = _evaluator(service, output_dir)
    if options.resume is not None and service is None:
        request = _require_same_agent(output_dir, options.resume, spec.ref.spec_sha256)
        options.judge = request.judge
        options.evaluation = request.evaluation  # what `{evaluation}` starts the agent with
        needs_model = _needs_model(request.evaluation, request.seed)
    hosted = connect(judge=options.judge and needs_model,
                     ask=_key_prompt() if _interactive() else None,
                     rerun=lambda: _no_judge_rerun(ctx))
    call.org = hosted.org
    try:
        return run_agent(evaluator, spec, options, output_dir, hosted, Instruments(trace=trace))
    finally:
        spec.connector.close()
        hosted.http.close()


def _evaluation_ref(given: str | None) -> str:
    """A shipped ref as given, or an evaluation file's absolute path (so a resume finds it)."""
    from memrank.definitions import is_file_ref

    if given is None:
        return ""
    return str(Path(given).resolve()) if is_file_ref(given) else given


def _needs_model(evaluation: str, seed: int) -> bool:
    """Whether the evaluation's grading calls the judge model, checking the whole file first."""
    from memrank.definitions import resolve as resolve_evaluation

    return resolve_evaluation(evaluation).uses_model(seed)


def _interactive() -> bool:
    """Whether someone at a terminal can answer a question. The seam tests patch."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_key(name: str) -> str:
    """Read a key with the input hidden; ``""`` when the user cancels with Ctrl-C or EOF."""
    try:
        return str(typer.prompt(name, hide_input=True, default="", show_default=False,
                                err=True))
    except typer.Abort:
        return ""


def _key_prompt():  # noqa: ANN202 - the loop's type, imported lazily like the rest of it
    from memrank.judging.client import key_refusal
    from memrank.loop.judge_key import KeyPrompt

    return KeyPrompt(read=_read_key, verify=key_refusal, echo=style.say)


def _given(ctx: typer.Context, skip: tuple[str, ...] = ()) -> list[str]:
    """The user's own ``memrank run`` arguments, as they typed them, minus ``skip``.

    Read from the command's own context, handed down, never from click's global one: Typer 0.27
    runs commands on a click it bundles, so the installed click has no current context at all.
    For the same reason the parameter source and kind are compared by name, not by class.
    """
    words = ["memrank", ctx.info_name or "run"]
    for param in ctx.command.params:
        name = param.name or ""
        source = ctx.get_parameter_source(name)
        if name in skip or source is None or source.name != "COMMANDLINE":
            continue
        value = ctx.params[name]
        if param.param_type_name == "argument":
            words.append(str(value))
        elif isinstance(value, bool):  # a --flag/--no-flag pair: the side that was typed
            words.append(param.opts[0] if value else param.secondary_opts[0])
        else:
            words += [param.opts[0], str(value)]
    return words


def _command(ctx: typer.Context) -> str:
    """The user's own command, to run again as it was."""
    return shlex.join(_given(ctx))


def _no_judge_rerun(ctx: typer.Context) -> str:
    """The user's own ``memrank run`` arguments, with ``--no-judge`` in place of judging."""
    return shlex.join([*_given(ctx, skip=("judge",)), "--no-judge"])


#: What a resumed run is given again besides its id: the rest is saved with the run.
_RESUME_KEEPS = ("agent", "output_dir", "service", "concurrency", "judge_model")


def _resume_command(ctx: typer.Context, run_id: str) -> str:
    """The command that continues ``run_id``, with the options it has to be given again."""
    kept = [param.name for param in ctx.command.params if param.name not in _RESUME_KEEPS]
    words = _given(ctx, skip=tuple(name or "" for name in kept))
    words[2:2] = ["--resume", run_id]
    return shlex.join(words)


def _evaluator(service: str | None, output_dir: Path):  # noqa: ANN202
    """The service in this process, or the client of a shared one."""
    if service is not None:
        from memrank.loop.client import ServiceClient

        return ServiceClient(service)
    from memrank.service.engine import EvaluationService
    from memrank.service.store import FolderRunStore

    return EvaluationService(store=FolderRunStore(output_dir))


def _require_same_agent(output_dir: Path, run_id: str, sha256: str | None):  # noqa: ANN202
    """Refuse to resume a run with an agent file that is not the one it started with.

    Returns the run's request: its evaluation, and whether it is judged, which a resume keeps.
    """
    from memrank.service.store import FolderRunStore

    request, _ = FolderRunStore(output_dir).load(run_id)
    if request.agent.spec_sha256 != sha256:
        started_with = request.agent.spec_path or "<agent file>"
        raise ActionRequired(
            f"Run {run_id} was started with a different agent file: {started_with} (sha256 "
            f"{request.agent.spec_sha256}). A run continues only with the agent it started "
            "with, so its answers all come from one agent.",
            steps=(Step("Resume it with that file:",
                        (shlex.join(["memrank", "run", "--resume", run_id, "--agent",
                                     started_with]),)),
                   Step("Or start a new run with this agent.")))
    return request


def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(DEFAULT_SERVICE_PORT, "--port"),
    runs_dir: Path = typer.Option(DEFAULT_OUTPUT, "--dir", help="Where run folders are kept"),
) -> None:
    """Share one evaluation service over HTTP (only for `memrank run --service URL`)."""
    from memrank.service.app import create_app
    from memrank.service.engine import EvaluationService
    from memrank.service.store import FolderRunStore

    uvicorn = optional_import("uvicorn", "service")
    service = EvaluationService(store=FolderRunStore(runs_dir))
    style.note(f"evaluation service on http://{host}:{port}, runs in {runs_dir}")
    uvicorn.run(create_app(service), host=host, port=port, log_level="warning")


@agents_app.command("ls")
def agents_ls() -> None:
    """List the agents memrank ships. Pass one's name to `memrank run --agent`."""
    from memrank.agents import shipped
    from memrank.connect.spec import load_agent

    for name, path in shipped().items():
        spec = load_agent(path)
        style.out(f"{name:<16} {spec.description or ''}")


@agents_app.command("serve")
def agents_serve(
    kind: str = typer.Argument(..., help="Which reference agent: full-context"),
    model: str = typer.Option(..., "--model", help="Anthropic model that answers, e.g. "
                                                    "claude-haiku-4-5"),
    evaluation: str = typer.Option(..., "--evaluation",
                                   help="The eval ref the agent answers (e.g. beam:100k); its "
                                        "official reader prompt is the one it answers with"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(DEFAULT_AGENT_PORT, "--port"),
) -> None:
    """Serve a reference agent on the OpenAI Chat Completions API. `memrank run` starts it."""
    from memrank.judging.client import JUDGE_MODEL_PREFIX, JUDGE_SECRET, build_completer

    if kind not in REFERENCE_AGENTS:
        raise typer.BadParameter(f"unknown reference agent {kind!r}; choose {REFERENCE_AGENTS}")
    if not model.startswith(JUDGE_MODEL_PREFIX):
        raise typer.BadParameter(f"the reference agent answers through memrank's Anthropic "
                                 f"reader; --model must start with {JUDGE_MODEL_PREFIX!r}")
    if not config.secret(JUDGE_SECRET):
        raise MemrankError(f"the {kind} agent answers with Anthropic: export {JUDGE_SECRET}")
    uvicorn = optional_import("uvicorn", "service")
    from memrank.benchmarks.answer_prompts import answer_prompt_for
    from memrank.reference.full_context import create_app

    prompt = answer_prompt_for(evaluation)
    complete, _ = build_completer(JudgeConfig(answer_model=model, cache=False))
    style.note(f"{kind} agent ({model}) on http://{host}:{port}/v1/chat/completions")
    # Printed to the agent log `memrank run` keeps in the run folder: which reader answered.
    style.note(f"{kind} agent answers {evaluation} with answer prompt {prompt.identity} "
               f"({prompt.source})")
    uvicorn.run(create_app(complete, model, prompt), host=host, port=port,
                log_level="warning")
