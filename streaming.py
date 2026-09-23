"""Hadoop Streaming entry points. Both map and reduce execute on Hadoop workers."""

import json
import os
import sys
from pathlib import Path

from pipeline import process


def mapper():
    input_path = os.environ.get("map_input_file") or os.environ.get("mapreduce_map_input_file", "")
    stem = Path(input_path).stem
    table = "uncertain" if stem == "unresolved_conflicts" else stem.removesuffix("_clean")
    if table not in {"users", "movies", "ratings", "uncertain"}:
        raise RuntimeError(f"Cannot identify input table from Hadoop path: {input_path!r}")
    encoding = "utf-8" if stem.endswith("_clean") or table == "uncertain" else "latin-1"
    for raw in sys.stdin.buffer:
        line = raw.decode(encoding).rstrip("\r\n")
        print("0\t" + json.dumps([table, line], ensure_ascii=False))


def reducer():
    phase = sys.argv[2] if len(sys.argv) > 2 else "clean"
    if phase not in {"before", "clean", "after"}:
        raise ValueError(f"Unknown phase: {phase}")
    def records():
        for line in sys.stdin:
            _, payload = line.rstrip("\n").split("\t", 1)
            yield json.loads(payload)

    def emit(kind, payload):
        if phase == "clean" or kind == "report":
            print(kind + "\t" + payload)

    process(records(), emit)


if __name__ == "__main__":
    {"mapper": mapper, "reducer": reducer}[sys.argv[1]]()
