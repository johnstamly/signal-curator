"""Content-hash code versioning of cells (ruling R18).

A cell kind declares the files whose content can change its result.  Its *code key* is the
sha256 over the sorted (path, bytes) of those files, with CRLF normalised to LF so line-ending
conversion (autocrlf) cannot change it.  Unrelated commits (latex/, other modules, new studies)
leave the key untouched; a change to any dependency file changes it.  ``dirty`` lists the
dependency files that differ from HEAD (``git status --porcelain``, untracked included).
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def code_key(files, root: Path) -> str:
    h = hashlib.sha256()
    for rel in sorted(set(files)):
        p = Path(root) / rel
        if not p.is_file():
            raise FileNotFoundError(f"dependency file missing: {rel}")
        h.update(rel.encode("utf-8") + b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def dirty_files(files, root: Path) -> list[str]:
    """Dependency files that differ from HEAD (modified, staged or untracked)."""
    files = sorted(set(files))
    out = subprocess.run(["git", "status", "--porcelain", "--", *files], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    return sorted(line[3:].strip().strip('"') for line in out.splitlines() if line.strip())
