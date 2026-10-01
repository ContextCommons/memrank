"""A command grader: full marks when the answer contains the reference, ignoring case."""

import json
import sys

given = json.load(sys.stdin)
hit = str(given["reference"]).lower() in given["answer"].lower()
print(json.dumps({"score": 1.0 if hit else 0.0,
                  "why": "mentions the reference" if hit else "does not mention it"}))
