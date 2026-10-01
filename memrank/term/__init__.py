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
"""Terminal presentation -- the one place memrank decides how output looks.

Everything a human reads on a terminal is shaped here and nowhere else. `style` owns the
stdout/stderr split and colour (a single chokepoint, so `NO_COLOR` and stream discipline are
enforced by construction rather than by convention). `fmt` turns raw values into readable ones --
durations, byte counts, rates, money -- and deliberately refuses to round scores, because a score
is a result, not a presentation detail. `table` lays out a listing and `detail` lays out a
`show` view, each rendering two ways -- bordered and coloured for a terminal, plain rows and
pairs for a pipe -- because a border is a human affordance and a pipe is not a human.
`progress` renders the progress model two ways on the same principle, redrawn in place for a
live TTY and append-only everywhere else. `key_input` reads a single keypress so a long
foreground command can be quit without a signal.

`rich` is imported by `table` and `detail` and by nothing else, and it renders into a buffer
rather than to a stream, so `style.out`/`style.say` remain the only emitters in the codebase.

Nothing in here computes anything. If a module needs to decide *what* a number is, that belongs
upstream in the domain; this package only decides how it is spelled.
"""
