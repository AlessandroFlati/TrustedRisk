"""Download (or verify) the FHIR Bundle fixtures used by the demo + tests.

The 3 fixtures (~94 MB total) are too large to ship in-tree, so they live
in a release artifact and are pulled on demand. The script is idempotent:
files already present + matching SHA-256 are left untouched.

Usage:
    python scripts/download_fixtures.py                # default: GitHub Release v0.8.0
    TRUSTEDRISK_FIXTURES_URL=https://... python scripts/download_fixtures.py
    python scripts/download_fixtures.py --verify-only  # no network, only check hashes

Environment:
    TRUSTEDRISK_FIXTURES_URL  Base URL hosting <name>.json. Default:
                              https://github.com/AlessandroFlati/TrustedRisk/releases/download/v0.8.0
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"

DEFAULT_BASE_URL = (
    "https://github.com/AlessandroFlati/TrustedRisk/releases/download/v0.8.0"
)

# name -> (sha256, expected_size_bytes)
FIXTURES: dict[str, tuple[str, int]] = {
    "patient_01_clean.json": (
        "25fe2ece4b53bca5199c5fa2544111102bebdb32f0cf8b51a49d38d5c29c9012",
        7_712_233,
    ),
    "patient_02_abstain.json": (
        "23cca743a9139ddd173706c39d9b38d104e555b1fd444b313f24b8906f2fe59a",
        31_562_456,
    ),
    "patient_03_complex.json": (
        "e81d3d2edf074927eb1f5ab5b76d00af89f7e3d88ac99098aaae2bea15e115e8",
        56_995_126,
    ),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_present_and_valid(name: str) -> bool:
    target = FIXTURES_DIR / name
    if not target.is_file():
        return False
    expected_hash, expected_size = FIXTURES[name]
    if target.stat().st_size != expected_size:
        return False
    return sha256_of(target) == expected_hash


def download(name: str, base_url: str) -> None:
    url = f"{base_url.rstrip('/')}/{name}"
    target = FIXTURES_DIR / name
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(
            f"ERROR: failed to download {url}: {exc}\n"
            f"Set TRUSTEDRISK_FIXTURES_URL to a reachable mirror, or place the\n"
            f"fixture files manually under {FIXTURES_DIR}/."
        ) from exc

    expected_hash, expected_size = FIXTURES[name]
    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        tmp.unlink()
        raise SystemExit(
            f"ERROR: size mismatch for {name}: got {actual_size} bytes, "
            f"expected {expected_size}."
        )
    actual_hash = sha256_of(tmp)
    if actual_hash != expected_hash:
        tmp.unlink()
        raise SystemExit(
            f"ERROR: SHA-256 mismatch for {name}: got {actual_hash}, "
            f"expected {expected_hash}."
        )
    tmp.replace(target)
    print(f"  OK ({actual_size:,} bytes, sha256 {actual_hash[:12]}...)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download anything; exit non-zero if any fixture is missing or bad.",
    )
    args = parser.parse_args()

    base_url = os.environ.get("TRUSTEDRISK_FIXTURES_URL", DEFAULT_BASE_URL)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fixtures dir: {FIXTURES_DIR}")
    print(f"Source URL:   {base_url}")
    print()

    missing: list[str] = []
    ok: list[str] = []
    for name in FIXTURES:
        if is_present_and_valid(name):
            ok.append(name)
            print(f"  [OK]      {name} (already present, hash matches)")
        else:
            missing.append(name)
            print(f"  [MISSING] {name}")

    if not missing:
        print(f"\nAll {len(ok)} fixtures present and valid. Nothing to do.")
        return 0

    if args.verify_only:
        print(f"\n{len(missing)} fixture(s) missing or invalid:", *missing, sep="\n  - ")
        return 1

    print(f"\nDownloading {len(missing)} fixture(s)...")
    for name in missing:
        download(name, base_url)

    print(f"\nDone. {len(missing)} fixture(s) downloaded, {len(ok)} already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
