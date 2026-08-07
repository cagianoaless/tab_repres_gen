#!/usr/bin/env python3
"""Build the artifact checksum manifest from Git-tracked publication files."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        Path(value.decode("utf-8"))
        for value in completed.stdout.split(b"\0")
        if value and value != b"MANIFEST.sha256"
    )


def main() -> None:
    lines = []
    for relative in tracked_files():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative.as_posix()}")
    temporary = MANIFEST.with_suffix(".sha256.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)
    print(f"Wrote {len(lines)} checksums to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
