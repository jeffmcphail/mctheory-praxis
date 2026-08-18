"""
split_zip.py
============
Three-in-one repo archival tool for uploading context to Claude.

Modes
-----
  zip     Create a zip of the repo, excluding everything in .gitignore,
          the .git directory, .env files, and any .zip files in the repo root.
          With --repo-delta, zips ONLY the files that changed since a brief
          landed (see "Delta mode" below).
  split   Split an existing large zip into chunks < 90 MB each.
  bundle  Zip the repo *and* split in one step (zip + split).

Usage examples:
    # Zip only (produces ai-agent-factory.zip)
    python split_zip.py zip

    # Zip with a custom name
    python split_zip.py zip -o my_snapshot.zip

    # Delta: only what changed since the most recent brief landed
    python split_zip.py zip --repo-delta

    # Delta: only what changed since a specific brief landed
    python split_zip.py zip --repo-delta guard_matches_loader
    python split_zip.py zip --repo-delta --by-commit   Select by git diff instead of mtime

    # Split an existing zip
    python split_zip.py split ai-agent-factory.zip
    python split_zip.py split ai-agent-factory.zip --chunk-mb 50

    # Bundle: zip the repo then split
    python split_zip.py bundle
    python split_zip.py bundle --chunk-mb 50

Output (split / bundle):
    ai-agent-factory_part001.zip, ai-agent-factory_part002.zip, …
    ai-agent-factory_manifest.txt  — lists which files are in which chunk

Delta mode
----------
``--repo-delta [BRIEF_STEM]`` zips only the files that differ from the commit in
which a brief landed — i.e. exactly the work done in response to that brief.

  * No argument  → uses the most recent commit that ADDED a
                   ``claude/handoffs/BRIEF_*.md`` file.
  * BRIEF_STEM   → uses the commit that added
                   ``claude/handoffs/BRIEF_<stem>.md``. The ``BRIEF_`` prefix
                   and ``.md`` suffix are optional, so all of these work:
                       guard_matches_loader
                       BRIEF_guard_matches_loader
                       BRIEF_guard_matches_loader.md

Selection (default: file mtime) — every non-excluded file whose filesystem
modification time is newer than the brief commit. This captures work whether or
not it has been committed, and — importantly — ignores long-standing uncommitted
files that predate the brief.

``--by-commit`` switches to ``git diff`` against the brief commit. That is exact
about content, but it also lists everything left uncommitted from *before* the
brief, since those differ from every commit. Use it when you want "what differs
from that commit", not "what was worked on since".

Output name defaults to ``<repo>_delta_<brief_stem>.zip`` so it never clobbers a
full snapshot.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# .gitignore parser
# ---------------------------------------------------------------------------

def _parse_gitignore(gitignore_path: Path) -> list[tuple[str, bool]]:
    """Return a list of (pattern, negated) tuples from a .gitignore file."""
    rules: list[tuple[str, bool]] = []
    if not gitignore_path.exists():
        return rules
    for raw_line in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        negated = False
        if line.startswith("!"):
            negated = True
            line = line[1:]
        line = line.rstrip()
        rules.append((line, negated))
    return rules


def _pattern_matches(pattern: str, rel_path: str, is_dir: bool) -> bool:
    """Check whether a single .gitignore pattern matches a relative path.

    ``rel_path`` uses forward slashes and has no leading slash.
    """
    dir_only = pattern.endswith("/")
    if dir_only:
        pattern = pattern.rstrip("/")
        if not is_dir:
            return False

    if "/" in pattern:
        anchored = pattern.lstrip("/")
        return _glob_match(anchored, rel_path)
    else:
        # Unanchored — matches any path component or basename
        parts = rel_path.split("/")
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
        return False


def _glob_match(pattern: str, path: str) -> bool:
    """fnmatch-style match with support for ** (recursive wildcard)."""
    regex = _glob_to_regex(pattern)
    return re.match(regex, path) is not None


def _glob_to_regex(pattern: str) -> str:
    """Convert a gitignore-style glob to a Python regex."""
    parts = pattern.split("**")
    converted = []
    for part in parts:
        seg = ""
        for ch in part:
            if ch == "*":
                seg += "[^/]*"
            elif ch == "?":
                seg += "[^/]"
            elif ch == "/":
                seg += "/"
            else:
                seg += re.escape(ch)
        converted.append(seg)
    regex = ".*".join(converted)
    return f"^{regex}(/.*)?$"


def _is_ignored(
    rel_path: str,
    is_dir: bool,
    rules: list[tuple[str, bool]],
) -> bool:
    """Apply .gitignore rules in order.  Last matching rule wins."""
    ignored = False
    for pattern, negated in rules:
        if _pattern_matches(pattern, rel_path, is_dir):
            ignored = not negated
    return ignored


# ---------------------------------------------------------------------------
# Zip the repo
# ---------------------------------------------------------------------------

def zip_repo(
    repo_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create a zip of the repo, respecting .gitignore.

    Also always excludes:
      - .git/
      - .env (secrets)
      - *.zip files in the repo root (old archives)
    """
    repo_dir = Path(repo_dir or ".").resolve()
    if not repo_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {repo_dir}")

    repo_name = repo_dir.name
    output_path = (
        Path(output_path).resolve()
        if output_path
        else (repo_dir / f"{repo_name}.zip")
    )

    gitignore_path = repo_dir / ".gitignore"
    rules = _parse_gitignore(gitignore_path)

    files_to_zip: list[tuple[Path, str]] = []  # (abs_path, archive_name)

    for root, dirs, files in os.walk(repo_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(repo_dir).as_posix()
        if rel_root == ".":
            rel_root = ""

        # Filter directories in-place so os.walk skips them
        filtered_dirs = []
        for d in dirs:
            rel_d = f"{rel_root}/{d}" if rel_root else d
            if d == ".git":
                continue
            if _is_ignored(rel_d, is_dir=True, rules=rules):
                continue
            filtered_dirs.append(d)
        dirs[:] = sorted(filtered_dirs)

        for f in sorted(files):
            abs_f = root_path / f
            rel_f = f"{rel_root}/{f}" if rel_root else f

            # Skip .zip files in the repo root
            if not rel_root and f.lower().endswith(".zip"):
                continue

            # Always skip .env files (secrets) even if not in .gitignore
            if f == ".env":
                continue

            if _is_ignored(rel_f, is_dir=False, rules=rules):
                continue

            archive_name = f"{repo_name}/{rel_f}"
            files_to_zip.append((abs_f, archive_name))

    print(f"Zipping {len(files_to_zip)} files from {repo_dir.name}/ …")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_f, arc_name in files_to_zip:
            zf.write(abs_f, arc_name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Created {output_path.name}  ({size_mb:.1f} MB, {len(files_to_zip)} files)")
    return output_path


# ---------------------------------------------------------------------------
# Delta mode: only what changed since a brief landed
# ---------------------------------------------------------------------------

BRIEF_DIR = "claude/handoffs"
LATEST = "__latest__"


def _git(repo_dir: Path, *args: str) -> str:
    """Run a git command in repo_dir and return stdout (raises on failure)."""
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _normalise_brief_stem(stem: str) -> str:
    """Accept 'x', 'BRIEF_x' or 'BRIEF_x.md' and return the bare stem 'x'."""
    s = stem.strip()
    if s.endswith(".md"):
        s = s[:-3]
    if s.startswith("BRIEF_"):
        s = s[len("BRIEF_"):]
    return s


def _find_brief_commit(repo_dir: Path, brief_stem: str) -> tuple[str, str]:
    """Return (commit_sha, brief_filename) for the commit that ADDED a brief.

    ``brief_stem`` is either ``LATEST`` (most recent brief added) or a specific
    stem. Walks ``git log --diff-filter=A`` over the handoffs directory, which
    reports only commits that *introduced* files.
    """
    out = _git(
        repo_dir, "log", "--diff-filter=A", "--format=__C__%H", "--name-only",
        "--", BRIEF_DIR,
    )
    current: str | None = None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("__C__"):
            current = line[len("__C__"):]
            continue
        name = Path(line).name
        if not (name.startswith("BRIEF_") and name.endswith(".md")):
            continue
        if brief_stem == LATEST:
            return current, name  # log is newest-first
        if _normalise_brief_stem(name) == brief_stem:
            return current, name

    if brief_stem == LATEST:
        raise RuntimeError(
            f"No commit found that added a {BRIEF_DIR}/BRIEF_*.md file."
        )
    raise RuntimeError(
        f"No commit found that added {BRIEF_DIR}/BRIEF_{brief_stem}.md — "
        "check the stem, or that the brief has been committed."
    )


def _changed_since(repo_dir: Path, commit: str) -> tuple[list[str], list[str]]:
    """Return (existing_paths, deleted_paths) relative to the repo root.

    Covers tracked changes since ``commit`` (committed AND uncommitted, because
    a bare ``git diff <commit>`` compares the working tree) plus untracked files
    that .gitignore does not exclude.
    """
    tracked = _git(repo_dir, "diff", "--name-only", commit).splitlines()
    untracked = _git(
        repo_dir, "ls-files", "--others", "--exclude-standard"
    ).splitlines()

    existing, deleted = [], []
    for rel in sorted({p.strip() for p in tracked + untracked if p.strip()}):
        if (repo_dir / rel).is_file():
            existing.append(rel)
        else:
            deleted.append(rel)
    return existing, deleted


def _walk_repo_files(repo_dir: Path, rules: list[tuple[str, bool]]):
    """Yield (abs_path, rel_posix) for every non-excluded file in the repo.

    Same exclusions as :func:`zip_repo`: .gitignore, .git/, .env, root *.zip.
    """
    for root, dirs, files in os.walk(repo_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(repo_dir).as_posix()
        if rel_root == ".":
            rel_root = ""

        keep = []
        for d in dirs:
            rel_d = f"{rel_root}/{d}" if rel_root else d
            if d == ".git":
                continue
            if _is_ignored(rel_d, is_dir=True, rules=rules):
                continue
            keep.append(d)
        dirs[:] = sorted(keep)

        for fname in sorted(files):
            rel_f = f"{rel_root}/{fname}" if rel_root else fname
            if not rel_root and fname.lower().endswith(".zip"):
                continue
            if fname == ".env":
                continue
            if _is_ignored(rel_f, is_dir=False, rules=rules):
                continue
            yield root_path / fname, rel_f


def _files_modified_since(
    repo_dir: Path, since_epoch: float, rules: list[tuple[str, bool]]
) -> list[str]:
    """Relative paths whose filesystem mtime is newer than ``since_epoch``."""
    out = []
    for abs_f, rel_f in _walk_repo_files(repo_dir, rules):
        try:
            if abs_f.stat().st_mtime > since_epoch:
                out.append(rel_f)
        except OSError:
            continue
    return sorted(out)


def zip_repo_delta(
    repo_dir: str | Path | None = None,
    brief_stem: str = LATEST,
    output_path: str | Path | None = None,
    by_commit: bool = False,
) -> Path | None:
    """Zip only the files that changed since a brief landed.

    Two selection modes:

    * **mtime (default)** — files whose filesystem modification time is newer
      than the brief commit's timestamp. This is what you want in practice: it
      captures work whether or not it has been committed, and it ignores
      long-standing uncommitted residue that predates the brief.
    * **commit** (``by_commit=True``) — ``git diff`` against the brief commit.
      Precise about *content* changes, but it also reports every file that has
      been sitting uncommitted since before the brief, because those differ
      from every commit.

    Applies the same exclusions as :func:`zip_repo` (.gitignore, .env, root
    *.zip). Returns the archive path, or ``None`` when nothing changed.
    """
    repo_dir = Path(repo_dir or ".").resolve()
    if not (repo_dir / ".git").exists():
        raise RuntimeError(
            f"{repo_dir} is not a git repository — delta mode needs git history."
        )

    stem = LATEST if brief_stem == LATEST else _normalise_brief_stem(brief_stem)
    commit, brief_name = _find_brief_commit(repo_dir, stem)
    subject = _git(repo_dir, "log", "-1", "--format=%s", commit).strip()
    when = _git(repo_dir, "log", "-1", "--format=%ci", commit).strip()

    mode = "commit diff" if by_commit else "file mtime"
    print(f"Brief   : {brief_name}")
    print(f"Landed  : {commit[:9]}  {when}")
    print(f"          {subject}")
    print(f"Select  : {mode}")

    rules = _parse_gitignore(repo_dir / ".gitignore")
    repo_name = repo_dir.name
    deleted: list[str] = []

    if by_commit:
        existing, deleted = _changed_since(repo_dir, commit)
        existing = [
            rel for rel in existing
            if Path(rel).name != ".env"
            and not ("/" not in rel and rel.lower().endswith(".zip"))
            and not _is_ignored(rel, is_dir=False, rules=rules)
        ]
    else:
        epoch = float(_git(repo_dir, "log", "-1", "--format=%ct", commit).strip())
        existing = _files_modified_since(repo_dir, epoch, rules)

    files_to_zip: list[tuple[Path, str]] = [
        (repo_dir / rel, f"{repo_name}/{rel}") for rel in existing
    ]

    if deleted:
        print(f"\nDeleted since the brief ({len(deleted)}, not archived):")
        for rel in deleted:
            print(f"  - {rel}")

    if not files_to_zip:
        print("\nNo changed files to archive.")
        return None

    out_stem = brief_name[len("BRIEF_"):-3] if brief_name.startswith("BRIEF_") else "latest"
    output_path = (
        Path(output_path).resolve()
        if output_path
        else (repo_dir / f"{repo_name}_delta_{out_stem}.zip")
    )

    print(f"\nZipping {len(files_to_zip)} changed file(s) …")
    for _, arc in files_to_zip:
        print(f"  + {arc.split('/', 1)[1]}")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_f, arc_name in files_to_zip:
            zf.write(abs_f, arc_name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n  Created {output_path.name}  "
          f"({size_mb:.2f} MB, {len(files_to_zip)} files)")
    return output_path


# ---------------------------------------------------------------------------
# Split a zip
# ---------------------------------------------------------------------------

def split_zip(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    chunk_mb: float = 50.0,
) -> list[Path]:
    """Split a large zip into chunks that are each ≤ chunk_mb."""
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    chunk_bytes = int(chunk_mb * 1024 * 1024)
    stem = input_path.stem

    out_dir = Path(output_dir).resolve() if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read all entries ----
    print(f"Reading {input_path.name} …")
    with zipfile.ZipFile(input_path, "r") as zf:
        all_entries = zf.infolist()
        total = len(all_entries)
        print(f"  {total} entries, "
              f"{input_path.stat().st_size / 1024 / 1024:.1f} MB total")

    # ---- Sort entries by size descending so big files spread evenly ----
    all_entries.sort(key=lambda e: e.file_size, reverse=True)

    # ---- Bin-pack into chunks ----
    chunks: list[list[zipfile.ZipInfo]] = []
    chunk_sizes: list[int] = []

    for entry in all_entries:
        entry_size = entry.compress_size or entry.file_size or 0

        placed = False
        for i, (chunk, csize) in enumerate(zip(chunks, chunk_sizes)):
            if csize + entry_size <= chunk_bytes:
                chunk.append(entry)
                chunk_sizes[i] += entry_size
                placed = True
                break

        if not placed:
            chunks.append([entry])
            chunk_sizes.append(entry_size)

    print(f"  Splitting into {len(chunks)} chunk(s) of ≤ {chunk_mb} MB each")

    # ---- Write chunks ----
    output_paths: list[Path] = []
    manifest_lines = [
        f"Source: {input_path.name}",
        f"Total entries: {total}",
        f"Chunks: {len(chunks)}",
        "",
    ]

    with zipfile.ZipFile(input_path, "r") as src_zf:
        for i, chunk_entries in enumerate(chunks, start=1):
            part_name = f"{stem}_part{i:03d}.zip"
            part_path = out_dir / part_name

            manifest_lines.append(
                f"=== {part_name} ({len(chunk_entries)} files) ==="
            )

            with zipfile.ZipFile(
                part_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as dst_zf:
                for entry in chunk_entries:
                    data = src_zf.read(entry.filename)
                    dst_zf.writestr(entry, data)
                    manifest_lines.append(f"  {entry.filename}")

            size_mb = part_path.stat().st_size / 1024 / 1024
            print(
                f"  Wrote {part_name}  "
                f"({len(chunk_entries)} files, {size_mb:.1f} MB)"
            )
            manifest_lines.append(f"  -> {size_mb:.1f} MB")
            manifest_lines.append("")
            output_paths.append(part_path)

    # ---- Write manifest ----
    manifest_path = out_dir / f"{stem}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"\nManifest written to: {manifest_path.name}")
    print("Done. Upload each part zip to Claude separately.")
    print("Upload the manifest first so Claude knows the full structure.")

    return output_paths


# ---------------------------------------------------------------------------
# Bundle = zip + split
# ---------------------------------------------------------------------------

def bundle(
    repo_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    chunk_mb: float = 50.0,
) -> list[Path]:
    """Zip the repo then split into uploadable chunks."""
    repo_dir = Path(repo_dir or ".").resolve()
    zip_path = zip_repo(repo_dir=repo_dir)
    print()
    parts = split_zip(
        input_path=zip_path,
        output_dir=output_dir,
        chunk_mb=chunk_mb,
    )
    return parts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repo zip / split / bundle tool for Claude uploads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python split_zip.py zip                     Zip the repo (produces <repo>.zip)
  python split_zip.py zip -o snap.zip         Zip with custom output name\n  python split_zip.py zip --repo-delta        Only files changed since the latest brief\n  python split_zip.py zip --repo-delta guard_matches_loader
  python split_zip.py split ai-agent-factory.zip        Split an existing zip
  python split_zip.py split ai-agent-factory.zip --chunk-mb 40
  python split_zip.py bundle                  Zip + split in one step
  python split_zip.py bundle --chunk-mb 40
""",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- zip ----
    p_zip = sub.add_parser("zip", help="Zip the repo (respects .gitignore)")
    p_zip.add_argument(
        "--repo-dir", type=str, default=None,
        help="Repo root directory (default: current directory)",
    )
    p_zip.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output zip path (default: <repo>.zip in repo root)",
    )
    p_zip.add_argument(
        "--by-commit", action="store_true",
        help=(
            "With --repo-delta, select files by git diff against the brief "
            "commit instead of by file mtime. Note this also picks up anything "
            "left uncommitted from before the brief."
        ),
    )
    p_zip.add_argument(
        "--repo-delta", nargs="?", const=LATEST, default=None, metavar="BRIEF_STEM",
        help=(
            "Zip only files changed since a brief landed. With no value, uses "
            "the most recent committed BRIEF_*.md; pass a stem (e.g. "
            "guard_matches_loader) to diff from that brief instead."
        ),
    )

    # ---- split ----
    p_split = sub.add_parser("split", help="Split a zip into uploadable chunks")
    p_split.add_argument("zip_file", help="Path to the zip file to split")
    p_split.add_argument(
        "--chunk-mb", type=float, default=50.0,
        help="Max size per chunk in MB (default: 50)",
    )
    p_split.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output chunks (default: same as input)",
    )

    # ---- bundle ----
    p_bundle = sub.add_parser(
        "bundle", help="Zip the repo then split (zip + split)"
    )
    p_bundle.add_argument(
        "--repo-dir", type=str, default=None,
        help="Repo root directory (default: current directory)",
    )
    p_bundle.add_argument(
        "--chunk-mb", type=float, default=50.0,
        help="Max size per chunk in MB (default: 50)",
    )
    p_bundle.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output chunks (default: repo root)",
    )

    args = parser.parse_args()

    if args.command is None:
        # Default to "zip" when no subcommand is given
        args.command = "zip"
        args.repo_dir = None
        args.output = None
        args.repo_delta = None
        args.by_commit = False

    if args.command == "zip":
        if getattr(args, "repo_delta", None) is not None:
            try:
                zip_repo_delta(
                    repo_dir=args.repo_dir,
                    brief_stem=args.repo_delta,
                    output_path=args.output,
                    by_commit=getattr(args, "by_commit", False),
                )
            except RuntimeError as e:
                # Operator-facing failure (no such brief, not a git repo, git
                # not on PATH) — a message, not a traceback.
                raise SystemExit(f"error: {e}")
        else:
            zip_repo(
                repo_dir=args.repo_dir,
                output_path=args.output,
            )
    elif args.command == "split":
        split_zip(
            input_path=args.zip_file,
            output_dir=args.output_dir,
            chunk_mb=args.chunk_mb,
        )
    elif args.command == "bundle":
        bundle(
            repo_dir=args.repo_dir,
            output_dir=args.output_dir,
            chunk_mb=args.chunk_mb,
        )


if __name__ == "__main__":
    main()
