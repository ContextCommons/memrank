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
"""Accounts, client side: how the CLI authenticates and carries credentials.

`credentials`, `login_flow`, `loopback`, `pkce` and `run_secrets` are HTTP-client, OS-keyring and
PKCE code. They ship with `pip install memrank` and never touch a database -- the CLI reaches the
platform through the API like any other client.

The service half lives in :mod:`memrank.accounts.server`: the Postgres stores, SES sending, SSM
secrets and the operator CLI. Nothing in this module imports it.
"""
