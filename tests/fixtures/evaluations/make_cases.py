"""A case program: prints --count cases as JSON lines, varied by --seed."""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
for n in range(args.count):
    number = args.seed * 100 + n
    print(json.dumps({
        "id": f"n{n}",
        "history": [{"turns": [{"role": "user", "text": f"My locker number is {number}."}]}],
        "questions": [{"question": "What is my locker number?", "answer": str(number)}],
    }))
