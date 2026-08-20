r"""Back up the publication record, and be able to prove it has not changed.

WHY THIS EXISTS
  data/results holds the evidence for the R&D challenge: the k-Wave +20 deg baseline, our
  channel data, the extracted reference cases, every published figure and the animations. It
  is git-tracked, which protects against losing it - but not against the failure that actually
  worries me, which is a run or a script quietly REWRITING one of those files. Git would show
  a diff nobody looks at on a 42 MB binary, and the number in a document would no longer match
  the file it came from.

  So this does two separate jobs:
    ARCHIVE - a copy outside the repository, so a bad command in the working tree cannot
              reach it. Plain files, not a container format, because in five years the thing
              most likely to be missing is the tool needed to open a clever archive.
    MANIFEST - SHA-256 of every file, stored with the archive AND in the repo. That turns
              "did anything change?" from a hope into a check that takes ten seconds.

  Restoring is deliberately manual. Anything that can automatically overwrite the working
  tree is a new way to destroy it.

WHAT COUNTS AS THE RECORD
  Everything git tracks under presentation/. That folder holds only evidence - routine run
  output goes to data/, which is not version-controlled at all - so the structure answers
  "what is worth keeping" without a list to maintain.

USAGE
  python3 tools/publication_backup.py --backup            # copy + write manifests
  python3 tools/publication_backup.py --verify            # working tree vs the manifest
  python3 tools/publication_backup.py --verify-archive    # the archive vs the manifest

  The default destination is a sibling of the repository, overridable with --dest or
  $DVFEA_BACKUP. It is deliberately NOT inside data/, and not in the read-only Dropbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Deliberately NOT importing lib.paths: this runs on the HOST, because the destination is
# outside every container mount by design. A backup reachable from inside the container is a
# backup a container job can destroy.
REPO = Path(__file__).resolve().parents[3]
MANIFEST_IN_REPO = REPO / "presentation" / "MANIFEST.json"
DEFAULT_DEST = REPO.parent / "FEA-publication-backup"

# What counts as the publication record: everything git tracks under presentation/.
#
# That folder exists to hold evidence and nothing else - routine run output goes to data/,
# which is not version-controlled at all. So the directory boundary answers "what is worth
# keeping" and there is no list to maintain. An earlier version of this tool did maintain
# one, because data/results held both kinds of file and a pattern could not tell them apart.
SUBTREE = "presentation"
MANIFEST_IN_REPO_NOTE = "presentation/MANIFEST.json"


def tracked_files() -> list[Path]:
    """Everything git tracks under presentation/.

    Back to git as the source of truth, and correctly this time. An explicit allowlist was
    needed while data/results mixed publication data with routine run output, so "what is
    worth keeping" had to be enumerated file by file. The presentation/ folder answers that
    structurally: nothing routine is written there, so everything tracked there is evidence.
    """
    out = subprocess.run(["git", "ls-files", "-z", SUBTREE], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [REPO / q for q in out.split(chr(0)) if q]


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(files: list[Path]) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    entries = {}
    total = 0
    for i, f in enumerate(sorted(files), 1):
        rel = f.relative_to(REPO).as_posix()
        size = f.stat().st_size
        entries[rel] = {"sha256": sha256(f), "bytes": size}
        total += size
        if i % 20 == 0 or i == len(files):
            print(f"  hashed {i}/{len(files)}  ({total / 1e6:.0f} MB)", flush=True)
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit,
            "subtree": SUBTREE, "file_count": len(entries), "total_bytes": total,
            "files": entries}


def do_backup(dest: Path) -> int:
    files = tracked_files()
    if not files:
        sys.exit(f"nothing tracked under {SUBTREE} - refusing to write an empty backup")
    print(f"publication record: {len(files)} tracked files under {SUBTREE}")
    man = build_manifest(files)

    stamp = time.strftime("%Y%m%d")
    root = dest / f"publication_{stamp}"
    if root.exists():
        # Never silently merge into an existing snapshot: a half-overwritten backup is worse
        # than no backup, because it still looks like one.
        sys.exit(f"{root} already exists. Remove it, or pass --dest elsewhere.")
    print(f"\ncopying to {root}")
    for i, rel in enumerate(sorted(man["files"]), 1):
        src, dst = REPO / rel, root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if i % 20 == 0 or i == len(man["files"]):
            print(f"  copied {i}/{len(man['files'])}", flush=True)

    (root / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    MANIFEST_IN_REPO.write_text(json.dumps(man, indent=1), encoding="utf-8")
    readme = root / "README.txt"
    readme.write_text(
        "Publication record for the FEA vs k-Wave R&D challenge.\n\n"
        f"Snapshot taken {man['created']} at commit {man['commit'][:12]}.\n"
        f"{man['file_count']} files, {man['total_bytes'] / 1e6:.0f} MB.\n\n"
        "MANIFEST.json holds a SHA-256 for every file. The same manifest is committed in the\n"
        "repository at data/PUBLICATION_MANIFEST.json, so the working tree can be checked\n"
        "against it without this archive being present:\n\n"
        "    ./run.ps1 python3 tools/publication_backup.py --verify\n\n"
        "Restoring is deliberately manual - copy back the files you mean to restore. Nothing\n"
        "here overwrites a working tree automatically, because a tool that can do that is a\n"
        "new way to lose the record.\n", encoding="utf-8")

    print(f"\n{man['file_count']} files, {man['total_bytes'] / 1e6:.0f} MB")
    print(f"archive   {root}")
    print(f"manifest  {MANIFEST_IN_REPO.relative_to(REPO)}  (commit this)")
    return 0


def do_verify(root: Path | None) -> int:
    """Compare either the working tree or an archive against the committed manifest."""
    if not MANIFEST_IN_REPO.is_file():
        sys.exit(f"no manifest at {MANIFEST_IN_REPO}; run --backup first")
    man = json.loads(MANIFEST_IN_REPO.read_text(encoding="utf-8"))
    base = root or REPO
    what = "archive" if root else "working tree"
    print(f"verifying {what} against the manifest of {man['created']} "
          f"(commit {man['commit'][:12]}, {man['file_count']} files)")

    missing, changed, ok = [], [], 0
    for rel, meta in sorted(man["files"].items()):
        p = base / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if p.stat().st_size != meta["bytes"] or sha256(p) != meta["sha256"]:
            changed.append(rel)
            continue
        ok += 1

    # Files that appeared since are not a problem - new results are the point of the project.
    # Only loss and mutation matter.
    print(f"\n  unchanged {ok}")
    print(f"  MISSING   {len(missing)}")
    print(f"  CHANGED   {len(changed)}")
    for rel in (missing + changed)[:20]:
        kind = "MISSING" if rel in missing else "CHANGED"
        print(f"    {kind}  {rel}")
    if missing or changed:
        print("\nFAIL - the publication record no longer matches the manifest.")
        print("Nothing has been repaired. Compare against the archive before doing anything.")
        return 1
    print("\nPASS - every file in the publication record is byte-identical.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--backup", action="store_true", help="copy the record and write manifests")
    g.add_argument("--verify", action="store_true", help="check the WORKING TREE")
    g.add_argument("--verify-archive", action="store_true", help="check the ARCHIVE")
    ap.add_argument("--dest", default=os.environ.get("DVFEA_BACKUP") or str(DEFAULT_DEST),
                    help=f"where archives live (default {DEFAULT_DEST})")
    args = ap.parse_args()

    dest = Path(args.dest)
    if args.backup:
        sys.exit(do_backup(dest))
    if args.verify:
        sys.exit(do_verify(None))
    snaps = sorted(dest.glob("publication_*")) if dest.is_dir() else []
    if not snaps:
        sys.exit(f"no archive found under {dest}")
    sys.exit(do_verify(snaps[-1]))


if __name__ == "__main__":
    main()
