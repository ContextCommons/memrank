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
"""Numbers in the units people read -- and the ones deliberately left alone.

A real run printed this:

    ingest total sec    275.204
    corpus bytes        100844
    corpus tokens       27708

which is 4m 35s, 98.5 KB and 27,708 tokens. Nobody should have to do that conversion to read
their own results.

WHAT IS NOT CONVERTED, and why. A benchmark's SCORES are its product. ``composite`` 0.269737 and
``est_dollars_per_query`` 6.615e-05 pass through at full precision, because two runs differing in
the fifth decimal are two different results and a tidier rendering would make them look identical.
Rounding is for quantities whose magnitude is the point; measurement is not one of them.

The unit belongs to the VALUE, declared where the value is produced (`run_registry.cell_metrics`),
not inferred from a label here. Matching `"total sec"` to pick a formatter would put one fact in
two places, which is how a renamed label silently starts printing seconds as bytes.
"""
from __future__ import annotations

#: Rendered where a number is genuinely unknown. Never a zero, which reads as a measurement.
UNKNOWN = "—"

#: What each of `cell_metrics`' values is, so the renderer formats rather than guesses.
KINDS = ("seconds", "bytes", "count", "rate", "score", "money", "text")


def duration(seconds: float | None) -> str:
    """``4m 35s``, ``2h 05m``, ``45s`` -- a magnitude, not a stopwatch.

    Deliberately coarse above the minute: these are usually projections from an observed rate, and
    ``4m 35.204s`` would claim a precision the estimate does not have.
    """
    if seconds is None:
        return UNKNOWN
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def approx_duration(seconds: float | None) -> str:
    """A duration marked as an estimate: ``~4m``. Used where the number is a projection.

    Coarser than :func:`duration` on purpose -- an ETA that reads ``~4m 35s`` invites the reader to
    check it against a clock, which is not a promise a rate-based projection can keep.
    """
    if seconds is None:
        return UNKNOWN
    total = int(seconds)
    if total < 90:
        return f"~{total}s"
    minutes = total // 60
    if minutes < 60:
        return f"~{minutes}m"
    return f"~{minutes // 60}h {minutes % 60:02d}m"


def size(num_bytes: float | None) -> str:
    """``98.5 KB``, ``20.2 MB``, ``342 B`` -- decimal units, matching how S3 and `ls -h` report."""
    if num_bytes is None:
        return UNKNOWN
    value = float(num_bytes)
    if value < 1000:
        return f"{int(value)} B"
    for unit in ("KB", "MB", "GB"):
        value /= 1000
        if value < 1000 or unit == "GB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} GB"


def count(number: float | None) -> str:
    """``27,708`` -- thousands separated, because the digit count is what a reader is judging."""
    if number is None:
        return UNKNOWN
    return f"{int(number):,}"


def rate(per_second: float | None, noun: str = "item") -> str:
    """A rate on whichever side of one-per-second it falls.

    ``5.2 s/doc`` and ``3.4/s`` describe the same kind of quantity; ingest and retrieve differ by
    ~18x on hindsight, so forcing one direction makes one of them unreadable (``0.19 docs/s``).
    """
    if not per_second:
        return UNKNOWN
    if per_second >= 1000:
        # A no-op stage (`no-context`, `fixed-context`) clocks six figures; a decimal place there is noise.
        return f"{per_second:,.0f}/s"
    if per_second >= 1:
        return f"{per_second:.1f}/s"
    return f"{1 / per_second:.1f} s/{noun}"


def measurement(value: float | None) -> str:
    """A score, a price, a latency -- rendered WITHOUT rounding.

    ``:.6g`` keeps 6.615e-05 legible without padding 216.0 into scientific notation, and keeps
    0.269737 whole. This is the function that must not get tidier.
    """
    if value is None:
        return UNKNOWN
    return f"{value:.6g}"


def value(raw, kind: str, *, noun: str = "item") -> str:
    """Render ``raw`` according to its declared kind. The renderer's single entry point."""
    if raw is None:
        return UNKNOWN
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    if kind == "seconds":
        return duration(raw)
    if kind == "bytes":
        return size(raw)
    if kind == "count":
        return count(raw)
    if kind == "rate":
        return rate(raw, noun)
    if kind in ("score", "money"):
        return measurement(raw)
    return str(raw)
