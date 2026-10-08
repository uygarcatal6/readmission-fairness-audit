#!/usr/bin/env python3
"""
Re-download and verify the UCI "Diabetes 130-US Hospitals for Years 1999-2008"
dataset (UCI id 296, CC BY 4.0).

Primary method : direct download of the official UCI zip via HTTP.
Fallback       : `ucimlrepo` (pip install ucimlrepo), fetch_ucirepo(id=296).

The script writes into ./raw/ (relative to this file) and verifies SHA-256
checksums against the values recorded on 2026-09-16. Run:

    python download.py            # download (if missing) + verify
    python download.py --force    # re-download even if files exist

No personal or clinical data is contained in or produced by this script; the
UCI dataset is public and de-identified.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
ZIP_NAME = "diabetes+130-us+hospitals+for+years+1999-2008.zip"

RAW_DIR = Path(__file__).resolve().parent / "raw"

# SHA-256 checksums of the UCI files, recorded on 2026-09-16.
EXPECTED_SHA256 = {
    "diabetes+130-us+hospitals+for+years+1999-2008.zip":
        "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a",
    "diabetic_data.csv":
        "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    "IDS_mapping.csv":
        "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_via_http(dest_zip: Path) -> bool:
    """Try the direct UCI zip download. Returns True on success."""
    try:
        print(f"[http] GET {UCI_ZIP_URL}")
        req = Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest_zip.write_bytes(data)
        print(f"[http] wrote {dest_zip} ({len(data)} bytes)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[http] failed: {exc}", file=sys.stderr)
        return False


def download_via_ucimlrepo() -> bool:
    """Fallback: reconstruct the CSVs from the ucimlrepo package."""
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
    except ImportError:
        print("[ucimlrepo] not installed; run: pip install ucimlrepo",
              file=sys.stderr)
        return False
    try:
        print("[ucimlrepo] fetch_ucirepo(id=296)")
        repo = fetch_ucirepo(id=296)
        # Rebuild diabetic_data.csv (features + target, original column order).
        X = repo.data.features
        y = repo.data.targets
        ids = repo.data.ids
        import pandas as pd  # local import, only needed for fallback
        frame = pd.concat([ids, X, y], axis=1)
        out = RAW_DIR / "diabetic_data.csv"
        frame.to_csv(out, index=False)
        print(f"[ucimlrepo] wrote {out} (note: byte layout may differ from "
              f"the official zip; SHA-256 verification may not match)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[ucimlrepo] failed: {exc}", file=sys.stderr)
        return False


def verify() -> bool:
    ok = True
    for name, expected in EXPECTED_SHA256.items():
        p = RAW_DIR / name
        if not p.exists():
            print(f"[verify] MISSING {name}")
            ok = False
            continue
        actual = sha256(p)
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            ok = False
        print(f"[verify] {status:8} {name}  {actual}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if raw files already exist")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME
    csv_path = RAW_DIR / "diabetic_data.csv"

    need = args.force or not csv_path.exists()
    if need:
        got = download_via_http(zip_path)
        if got:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(RAW_DIR)
            print(f"[unzip] extracted into {RAW_DIR}")
        else:
            print("[main] HTTP failed, trying ucimlrepo fallback ...")
            if not download_via_ucimlrepo():
                print("[main] all download methods failed", file=sys.stderr)
                return 2
    else:
        print("[main] raw files present; skipping download (use --force)")

    ok = verify()
    if not ok:
        print("[main] verification reported problems (see above).",
              file=sys.stderr)
        return 1
    print("[main] all files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
