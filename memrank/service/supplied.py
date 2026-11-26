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
"""Prices a run is given for its own models: the agent file's ``prices:`` section.

memrank's table (:mod:`memrank.metrics.prices`) holds only prices a provider publishes. Any other
model -- a private router, a self-hosted embedder, a negotiated rate -- is priced by the user, who
may also change a price the table holds: a supplied price wins for its provider and model.

The accepted shape, checked here and again where the cost is computed:

- ``provider`` and ``model``: non-empty text, matched exactly against what the agent reports;
- ``input_per_mtok`` and ``output_per_mtok``: finite numbers of at least 0, US dollars per million
  tokens (an explicit 0 is a price; no price is never one);
- each provider and model at most once.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from memrank.errors import MemrankError
from memrank.metrics.prices import Price, PriceKey, supplied_price

#: USD per million tokens: a finite number of at least 0, never a bool or text.
Rate = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]


class SuppliedPrice(BaseModel):
    """One price the run was given, in US dollars per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_per_mtok: Rate
    output_per_mtok: Rate


class AppliedPrice(SuppliedPrice):
    """A supplied price the run's cost used, and whether it replaced the table's own row."""

    replaces_table: bool


class DuplicatePrice(MemrankError, ValueError):
    """The same provider and model was given two prices. A ValueError too, so a request model's
    validator reports it as a validation error."""


def supplied_table(prices: Iterable[SuppliedPrice]) -> dict[PriceKey, Price]:
    """The supplied prices as table rows, keyed like the table; each pair at most once."""
    table: dict[PriceKey, Price] = {}
    for given in prices:
        key = (given.provider, given.model)
        if key in table:
            raise DuplicatePrice(f"{given.provider}/{given.model} is given two prices")
        table[key] = supplied_price(given.input_per_mtok, given.output_per_mtok)
    return table
