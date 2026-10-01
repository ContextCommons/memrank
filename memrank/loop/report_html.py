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
"""``report.html``: one run, explained in plain language, in a single file with no external assets.

A reader who has never seen memrank should be able to answer from it: what was asked, of which
agent, how often it was right, how sure that number is, what failed, and what the agent said to
each question. Every value is escaped; nothing is fetched when the page opens.
"""

from __future__ import annotations

from html import escape

from memrank.service.protocol import Interval, QuestionOutcome, RunResult

_STYLE = """
body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;
color:#1b1b1b;background:#fff}
h1{font-size:1.5rem;margin-bottom:.2rem}h2{font-size:1.15rem;margin-top:2rem}
.muted{color:#5f5f5f}.big{font-size:2rem;font-weight:600}
.warn{border:2px solid #b3261e;background:#fdecea;padding:.8rem 1rem;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #ddd;padding:.35rem .5rem;text-align:left;vertical-align:top}
th{background:#f4f4f4}.num{text-align:right;white-space:nowrap}
.fail{color:#b3261e}.pass{color:#146c2e}
@media (prefers-color-scheme:dark){body{background:#121212;color:#e8e8e8}th{background:#222}
.muted{color:#a8a8a8}th,td{border-color:#333}.warn{background:#3b1512}}
"""

_METRICS = (
    ("Score", "The share of questions the judge marked correct (BEAM: the mean share of rubric "
              "points met). A question the agent failed on counts as wrong."),
    ("95% interval", "Where the score would likely fall if the same kind of cases were drawn "
                     "again. Computed by resampling whole cases, because questions about one "
                     "conversation are not independent. A wide interval means few cases."),
    ("Failure rate", "The share of questions with no answer because the agent errored, or its "
                     "history could not be loaded. Failures are the agent's, not the judge's."),
    ("Latency", "Time the agent took per call, measured by memrank around each request. "
                "p50 is the typical call; p95 is a slow one."),
)


def _pct(value: float | None) -> str:
    return "&ndash;" if value is None else f"{value:.1%}"


def _interval(value: Interval) -> str:
    low, high = value.ci95
    band = "" if low is None or high is None else f" (95% interval {low:.1%} to {high:.1%})"
    return f"{_pct(value.mean)}{band}, over {value.n} questions"


def _seconds(ms: float | None) -> str:
    return "&ndash;" if ms is None else f"{ms / 1000:.2f} s"


def _header(result: RunResult) -> str:
    who = result.agent
    version = f" {escape(who.version)}" if who.version else ""
    return (f"<h1>{escape(result.evaluation)} &times; {escape(who.name)}{version}</h1>"
            f"<p class=muted>Run {escape(result.run_id)} &middot; memrank "
            f"{escape(result.identity.memrank_version)}</p>")


def _headline(result: RunResult) -> str:
    if not result.judged:
        return (f"<p class=warn>{escape(result.notice or '')}</p><p>The table below lists every "
                "answer; none of them was graded, so this run says nothing about quality.</p>")
    return (f"<h2>How often the agent was right</h2><p class=big>{_pct(result.score.mean)}</p>"
            f"<p>{_interval(result.score)}. Judged by {escape(result.identity.judge_model or '')}"
            f"; {result.unjudged} answers the judge could not grade are left out.</p>")


def _categories(result: RunResult) -> str:
    if not result.per_category:
        return ""
    rows = "".join(f"<tr><td>{escape(name)}</td><td class=num>{_interval(value)}</td></tr>"
                   for name, value in result.per_category.items())
    return ("<h2>By kind of question</h2><table><tr><th>Category</th><th>Score</th></tr>"
            f"{rows}</table>")


def _reliability(result: RunResult) -> str:
    latency = result.latency_ms
    return (f"<h2>Failures and speed</h2><p>{result.failed} of {result.questions} questions "
            f"failed ({_pct(result.failure_rate)}). Answering took {_seconds(latency.get('ask_p50'))}"
            f" typically and {_seconds(latency.get('ask_p95'))} when slow; loading a case's "
            f"history took {_seconds(latency.get('feed_p50'))} typically.</p>")


def _explained() -> str:
    rows = "".join(f"<tr><th>{name}</th><td>{text}</td></tr>" for name, text in _METRICS)
    return f"<h2>What these numbers mean</h2><table>{rows}</table>"


def _identity(result: RunResult) -> str:
    who = result.identity
    agent = who.agent
    rows = (("Benchmark", f"{who.evaluation}, dataset {who.dataset_version}, task version "
                          f"{who.task_version}"),
            ("Sample", f"{len(who.case_ids)} cases (seed {who.seed}), at most "
                       f"{who.questions_per_case or 'all'} questions each: "
                       f"{', '.join(who.case_ids)}"),
            ("Agent file", f"{agent.spec_path} (sha256 {agent.spec_sha256})"),
            ("Judge", who.judge_model or "off"))
    body = "".join(f"<tr><th>{k}</th><td>{escape(str(v))}</td></tr>" for k, v in rows)
    return f"<h2>What was run</h2><table>{body}</table>"


def _verdict(q: QuestionOutcome) -> str:
    if q.status == "judged":
        mark = "pass" if q.passed else "fail"
        return f"<span class={mark}>{'correct' if q.passed else 'wrong'}</span> ({q.score:.2f})"
    if q.status == "failed":
        return f"<span class=fail>failed</span>: {escape(q.error or '')}"
    return escape(q.status.replace("_", " "))


def _row(q: QuestionOutcome) -> str:
    cells = (escape(q.case_id), escape(q.category or ""), escape(q.question),
             escape(q.answer or ""), escape(" | ".join(q.reference)), _verdict(q),
             _seconds(q.elapsed_ms))
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _questions(result: RunResult) -> str:
    rows = "".join(_row(q) for case in result.cases for q in case.questions)
    return ("<h2>Every question</h2><table id=questions><tr><th>Case</th><th>Category</th>"
            "<th>Question</th><th>Agent's answer</th><th>Reference</th><th>Verdict</th>"
            f"<th>Time</th></tr>{rows}</table>")


def render(result: RunResult) -> str:
    """The whole report as one HTML document."""
    body = "".join((_header(result), _headline(result), _categories(result),
                    _reliability(result), _explained(), _identity(result), _questions(result)))
    title = escape(f"{result.evaluation} x {result.agent.name}")
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{title}</title>"
            f"<style>{_STYLE}</style></head><body>{body}</body></html>\n")
