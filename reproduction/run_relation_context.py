"""Relocate the unchanged decoder launcher and record fresh analysis provenance."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from types import SimpleNamespace

from reproduction.verify_source_pipeline import HERE, load_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("decoders", "analyze"))
    parser.add_argument("--execution-source", help="actual Git SHA registered for a fresh run")
    args, remaining = parser.parse_known_args(argv)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    sys.path.insert(0, str(HERE / "study"))
    if args.stage == "decoders":
        if args.execution_source:
            parser.error("--execution-source is for analysis; workers record their actual HEAD")
        module = load_file(
            "_paper_decoder_launcher", HERE / "study/scripts/run_rfmlink_relation_context_remote.py"
        )

        def run_command(command, *positional, **keywords):
            command = list(command)
            if len(command) > 1 and command[1] == (
                "scripts/generate_rfmlink_relation_context_registration.py"
            ):
                command[1] = str(
                    HERE / "study/scripts/generate_rfmlink_relation_context_registration.py"
                )
            return subprocess.run(command, *positional, **keywords)

        # Adapt only the relocated generator command, not the worker/statistics.
        module.subprocess = SimpleNamespace(
            run=run_command,
            check_output=subprocess.check_output,
            DEVNULL=subprocess.DEVNULL,
            STDOUT=subprocess.STDOUT,
        )
    else:
        if not args.execution_source or not re.fullmatch(r"[0-9a-f]{40}", args.execution_source):
            parser.error("analysis requires --execution-source with the actual registered SHA")
        module = load_file(
            "_paper_relation_analysis", HERE / "study/scripts/analyze_rfmlink_relation_context.py"
        )
        # Fresh analyses use the execution SHA recorded by their registration.
        module.SOURCE_COMMIT = args.execution_source
    sys.argv = [f"relation-context {args.stage}", *remaining]
    result = module.main()
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
