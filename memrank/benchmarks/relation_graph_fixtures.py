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
"""Fixture pack for the relation-graph benchmark (data only).

Extracted from relation_graph.py to keep that module under the 400-line source
limit. Pure data: the relation-graph scenarios consumed by
RelationGraphBenchmark. No logic lives here.
"""
from __future__ import annotations

from typing import Any

FIXTURES: list[dict[str, Any]] = [
    {
        "id": "relation/ambiguous-update/001",
        "title": "Ambiguous Update With Similar Preference",
        "seed_memories": [
            {"id": "seed_avery_planning_blue", "text": "Avery prefers blue dashboards for planning reviews."},
            {"id": "seed_avery_demo_blue", "text": "Avery prefers blue dashboards for sprint demos."},
        ],
        "input_documents": [
            {
                "id": "doc_avery_planning_update",
                "text": (
                    "Avery is changing only the planning-review dashboard preference. "
                    "Avery now prefers green dashboards for planning reviews instead of blue dashboards. "
                    "Avery still prefers blue dashboards for sprint demos. Do not treat the sprint-demo "
                    "preference as replaced; only the planning-review preference is corrected."
                ),
            }
        ],
        "expected_memories": [
            {
                "id": "fact_avery_planning_green",
                "canonical_fact": "Avery prefers green dashboards for planning reviews.",
                "aliases": ["Avery now prefers green dashboards for planning reviews."],
                "required": True,
                "expected_inference": False,
                "relation_expectations": [
                    {"parent": "seed_avery_planning_blue", "relation": "updates", "required": True}
                ],
            }
        ],
        "expected_absences": [
            {"id": "absence_avery_demo_green", "fact": "Avery prefers green dashboards for sprint demos."}
        ],
        "queries": [
            {
                "id": "q_planning_preference",
                "text": "What dashboard color does Avery prefer for planning reviews?",
                "expected_retrieval": {
                    "required_facts": ["fact_avery_planning_green"],
                    "forbidden_facts": ["seed_avery_planning_blue"],
                    "required_context_relations": [
                        {
                            "fact": "fact_avery_planning_green",
                            "parent": "seed_avery_planning_blue",
                            "relation": "updates",
                        }
                    ],
                },
            },
            {
                "id": "q_demo_preference",
                "text": "What dashboard color does Avery prefer for sprint demos?",
                "expected_retrieval": {
                    "required_facts": ["seed_avery_demo_blue"],
                    "forbidden_facts": ["fact_avery_planning_green"],
                    "required_context_relations": [],
                },
            },
        ],
    },
    {
        "id": "relation/same-name-collision/001",
        "title": "Same-Name Collision With Project-Specific Extension",
        "seed_memories": [
            {"id": "seed_jordan_rivera_orion", "text": "Jordan Rivera works on Project Orion."},
            {"id": "seed_jordan_lee_lyra", "text": "Jordan Lee works on Project Lyra."},
        ],
        "input_documents": [
            {
                "id": "doc_jordan_lee_lyra_extension",
                "text": (
                    "Jordan Lee now uses Postgres as the local cache for Project Lyra. "
                    "Jordan Lee also tracks Project Lyra stale-chunk bugs in Linear. "
                    "Jordan Rivera still works only on Project Orion and should not be connected "
                    "to the Project Lyra facts."
                ),
            }
        ],
        "expected_memories": [
            {
                "id": "fact_jordan_lee_lyra_postgres",
                "canonical_fact": "Jordan Lee uses Postgres as the local cache for Project Lyra.",
                "aliases": [],
                "required": True,
                "expected_inference": False,
                "relation_expectations": [
                    {"parent": "seed_jordan_lee_lyra", "relation": "extends", "required": True}
                ],
            },
            {
                "id": "fact_jordan_lee_lyra_linear_bugs",
                "canonical_fact": "Jordan Lee tracks Project Lyra stale-chunk bugs in Linear.",
                "aliases": [],
                "required": True,
                "expected_inference": False,
                "relation_expectations": [
                    {"parent": "seed_jordan_lee_lyra", "relation": "extends", "required": True}
                ],
            },
            {
                "id": "fact_jordan_rivera_orion_only",
                "canonical_fact": "Jordan Rivera works only on Project Orion.",
                "aliases": [
                    "Jordan Rivera works only on Project Orion and should not be connected to Project Lyra facts."
                ],
                "required": False,
                "expected_inference": False,
                "relation_expectations": [],
            },
        ],
        "expected_absences": [
            {"id": "absence_rivera_lyra_postgres", "fact": "Jordan Rivera uses Postgres as the local cache for Project Lyra."},
            {"id": "absence_rivera_lyra_linear", "fact": "Jordan Rivera tracks Project Lyra stale-chunk bugs in Linear."},
        ],
        "queries": [
            {
                "id": "q_lyra_cache",
                "text": "What cache does Jordan Lee use for Project Lyra?",
                "expected_retrieval": {
                    "required_facts": ["fact_jordan_lee_lyra_postgres"],
                    "forbidden_facts": ["seed_jordan_rivera_orion"],
                    "required_context_relations": [
                        {
                            "fact": "fact_jordan_lee_lyra_postgres",
                            "parent": "seed_jordan_lee_lyra",
                            "relation": "extends",
                        }
                    ],
                },
            },
            {
                "id": "q_rivera_project",
                "text": "Which project does Jordan Rivera work on?",
                "expected_retrieval": {
                    "required_facts": ["fact_jordan_rivera_orion_only"],
                    "forbidden_facts": [
                        "fact_jordan_lee_lyra_postgres",
                        "fact_jordan_lee_lyra_linear_bugs",
                    ],
                    "required_context_relations": [],
                },
            },
        ],
    },
    {
        "id": "relation/messy-derives/001",
        "title": "Derivation With Distractor Report",
        "seed_memories": [
            {"id": "seed_mina_reports_omar", "text": "Mina reports to Omar."},
            {"id": "seed_omar_owns_ir", "text": "Omar owns the incident-response rotation."},
            {"id": "seed_nora_reports_omar", "text": "Nora reports to Omar."},
        ],
        "input_documents": [
            {
                "id": "doc_mina_escalation_delegate",
                "text": (
                    "Mina reports to Omar, and Omar owns the incident-response rotation. "
                    "Nora also reports to Omar, but Nora is not the escalation recipient in this memo. "
                    "Because Mina is the named delegate for this week and Mina reports to Omar, "
                    "incident-response escalation notes for this week should be sent to Mina."
                ),
            }
        ],
        "expected_memories": [
            {
                "id": "fact_mina_escalation_recipient",
                "canonical_fact": "Incident-response escalation notes for this week should be sent to Mina.",
                "aliases": ["Mina should receive incident-response escalation notes this week."],
                "required": True,
                "expected_inference": True,
                "relation_expectations": [
                    {"parent": "seed_mina_reports_omar", "relation": "derives", "required": True},
                    {"parent": "seed_omar_owns_ir", "relation": "derives", "required": True},
                ],
            }
        ],
        "expected_absences": [
            {"id": "absence_nora_escalation_notes", "fact": "Incident-response escalation notes for this week should be sent to Nora."},
            {"id": "absence_nora_escalation_recipient", "fact": "Nora is the escalation recipient."},
        ],
        "queries": [
            {
                "id": "q_escalation_recipient",
                "text": "Who should receive incident-response escalation notes this week and why?",
                "expected_retrieval": {
                    "required_facts": ["fact_mina_escalation_recipient"],
                    "forbidden_facts": ["seed_nora_reports_omar"],
                    "required_context_relations": [
                        {
                            "fact": "fact_mina_escalation_recipient",
                            "parent": "seed_mina_reports_omar",
                            "relation": "derives",
                        },
                        {
                            "fact": "fact_mina_escalation_recipient",
                            "parent": "seed_omar_owns_ir",
                            "relation": "derives",
                        },
                    ],
                },
            }
        ],
    },
    {
        "id": "relation/long-mixed/001",
        "title": "Two-Chunk Mixed Update Extension And Inference",
        "seed_memories": [
            {"id": "seed_priya_beacon", "text": "Priya works on Project Beacon."},
            {"id": "seed_priya_beacon_email", "text": "Priya prefers email summaries for Project Beacon."},
        ],
        "input_documents": [
            {
                "id": "doc_priya_beacon_long_mixed",
                "text": (
                    "Background: Project Beacon has unrelated operational notes. Priya continues to work on "
                    "Project Beacon. Project Beacon now uses a nightly reconciliation job for stale vector rows. "
                    "Project Beacon now stores ingest audit records for each document chunk. Priya previously "
                    "preferred email summaries for Project Beacon, but Priya now prefers Slack summaries for "
                    "Project Beacon instead. Because Project Beacon stores ingest audit records and Priya owns "
                    "the review workflow, Priya should receive the weekly ingest-audit exception report. "
                    "Distractor: Priya likes email for personal travel planning; that unrelated personal "
                    "preference is not about Project Beacon. Additional detail: stale vector rows are checked "
                    "at 02:00 UTC and exceptions are held for manual review."
                ),
            }
        ],
        "expected_memories": [
            {
                "id": "fact_priya_beacon_slack",
                "canonical_fact": "Priya prefers Slack summaries for Project Beacon.",
                "aliases": [
                    "Priya now prefers Slack summaries for Project Beacon.",
                    "Priya now prefers Slack summaries for Project Beacon instead of email summaries.",
                ],
                "required": True,
                "expected_inference": False,
                "relation_expectations": [
                    {"parent": "seed_priya_beacon_email", "relation": "updates", "required": True}
                ],
            },
            {
                "id": "fact_beacon_reconciliation",
                "canonical_fact": "Project Beacon uses a nightly reconciliation job for stale vector rows.",
                "aliases": [
                    "Project Beacon now uses a nightly reconciliation job for stale vector rows.",
                    "Project Beacon uses a nightly reconciliation job to check stale vector rows.",
                ],
                "required": True,
                "expected_inference": False,
                "relation_expectations": [
                    {"parent": "seed_priya_beacon", "relation": "extends", "required": True}
                ],
            },
            {
                "id": "fact_beacon_audit_records",
                "canonical_fact": "Project Beacon stores ingest audit records for each document chunk.",
                "aliases": ["Project Beacon now stores ingest audit records for each document chunk."],
                "required": True,
                "expected_inference": False,
                "relation_expectations": [
                    {"parent": "seed_priya_beacon", "relation": "extends", "required": True}
                ],
            },
            {
                "id": "fact_priya_ingest_audit_report",
                "canonical_fact": "Priya should receive the weekly ingest-audit exception report for Project Beacon.",
                "aliases": [
                    "Priya receives the weekly ingest-audit exception report for Project Beacon.",
                ],
                "required": True,
                "expected_inference": True,
                "relation_expectations": [
                    {"parent": "seed_priya_beacon", "relation": "derives", "required": True}
                ],
            },
        ],
        "expected_absences": [
            {"id": "absence_beacon_email_current", "fact": "Project Beacon prefers email summaries."},
        ],
        "queries": [
            {
                "id": "q_summary_channel",
                "text": "What summary channel does Priya prefer for Project Beacon now?",
                "expected_retrieval": {
                    "required_facts": ["fact_priya_beacon_slack"],
                    "forbidden_facts": ["seed_priya_beacon_email"],
                    "required_context_relations": [
                        {
                            "fact": "fact_priya_beacon_slack",
                            "parent": "seed_priya_beacon_email",
                            "relation": "updates",
                        }
                    ],
                },
            },
            {
                "id": "q_report_recipient",
                "text": "Who should receive the Project Beacon ingest-audit exception report?",
                "expected_retrieval": {
                    "required_facts": ["fact_priya_ingest_audit_report"],
                    "forbidden_facts": [],
                    "required_context_relations": [
                        {
                            "fact": "fact_priya_ingest_audit_report",
                            "parent": "seed_priya_beacon",
                            "relation": "derives",
                        }
                    ],
                },
            },
        ],
    },
]
