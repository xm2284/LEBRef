from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
import torch

import esra_core


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "PROTOCOL_V6.md",
    "README.md",
    "esra_core.py",
    "run_experiment.py",
    "run_cloud.sh",
    "extract_results.py",
    "test_core.py",
    "requirements_no_torch.txt",
    "configs/smartfallmm_fold_lock.json",
    "configs/umafall_fold_lock.json",
    "legacy_results/smartfallmm/fivefold_results.json",
    "legacy_results/umafall/fivefold_results.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required package files: {missing}")
    for dataset in ("smartfallmm", "umafall"):
        result_root = ROOT / "legacy_results" / dataset
        checkpoints = sorted(result_root.glob("fold*/seed*/best_head.pth"))
        records = sorted(result_root.glob("fold*/seed*/result.json"))
        if len(checkpoints) != 15 or len(records) != 15:
            raise RuntimeError(
                f"{dataset}: expected 15 legacy checkpoints and records, got "
                f"{len(checkpoints)} and {len(records)}"
            )
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            head = esra_core.new_head()
            head.load_state_dict(payload["head_state_dict"], strict=True)
            if sum(parameter.numel() for parameter in head.parameters()) != 2050:
                raise RuntimeError(f"unexpected head parameter count: {checkpoint}")
        lock = json.loads((ROOT / "configs" / f"{dataset}_fold_lock.json").read_text())
        if len(lock["folds"]) != 5:
            raise RuntimeError(f"{dataset}: fold lock does not contain five folds")
    print(
        json.dumps(
            {
                "required_files": len(REQUIRED),
                "legacy_heads": 30,
                "trainable_parameters": 2050,
                "protocol_sha256": sha256(ROOT / "PROTOCOL_V6.md"),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "gpu_available": torch.cuda.is_available(),
            },
            indent=2,
        )
    )
    print("FALL_ESRA_HEAD_V6_PACKAGE_OK")


if __name__ == "__main__":
    main()

