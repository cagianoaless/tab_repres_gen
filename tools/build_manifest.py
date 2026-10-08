#!/usr/bin/env python3
"""Build release checksums, including new non-ignored files before commit."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def release_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        {
            Path(value.decode("utf-8"))
            for value in completed.stdout.split(b"\0")
            if value and value != b"MANIFEST.sha256"
        }
    )


# Preserve existing imports; coverage now includes not-yet-staged release files.
tracked_files = release_files


def main() -> None:
    lines = []
    for relative in release_files():
        path = ROOT / relative
        if not path.is_file() or "\n" in str(relative) or "\r" in str(relative):
            raise ValueError(f"unsafe or missing release file: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative.as_posix()}")
    temporary = MANIFEST.with_suffix(".sha256.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)
    print(f"Wrote {len(lines)} checksums to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
