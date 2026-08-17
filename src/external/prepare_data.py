from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]

ARCHIVES = {
    "umafall": {
        "name": "umafall_corrected_20260723.zip",
        "sha256": "c952c3098b57d3fec8851aff89d95a462b4d52cd804d58cbccc10b6d5d2a1975",
        "target": "umafall_corrected",
        "csv_count": 746,
    },
    "smartfallmm": {
        "name": "smartfallmm_watch_selected_20260723.zip",
        "sha256": "96f1d4b13af4175eb3cd3561f90c522bb2ab0b17727fd31351b7581afb5d5735",
        "target": "smartfallmm_selected",
        "csv_count": 2407,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (target / member.filename).resolve()
            if destination != target_root and target_root not in destination.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        zipped.extractall(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing the locked dataset archives.",
    )
    args = parser.parse_args()
    for dataset, record in ARCHIVES.items():
        archive = args.workspace / str(record["name"])
        if not archive.is_file():
            raise FileNotFoundError(archive)
        actual = sha256_file(archive)
        if actual != record["sha256"]:
            raise RuntimeError(
                f"{archive.name}: SHA-256 mismatch: {actual} != {record['sha256']}"
            )
        target = args.workspace / str(record["target"])
        safe_extract(archive, target)
        observed = len(list(target.rglob("*.csv")))
        if observed != record["csv_count"]:
            raise RuntimeError(
                f"{dataset}: expected {record['csv_count']} CSV files, found {observed}"
            )
        print(f"{dataset}: archive_sha256={actual} csv_files={observed} target={target}")
    print("LOCKED_EXTERNAL_DATA_READY")


if __name__ == "__main__":
    main()
