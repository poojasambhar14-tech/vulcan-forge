import os
import sys

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure relative paths like "configs/tiny.yaml" resolve correctly and the
# `vulcan` package is importable regardless of the invocation directory.
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def trained_checkpoints():
    """Skips a test when the trained model artifacts are absent.

    Checkpoints are gitignored, so a fresh clone has none until
    train.pretrain and train.train_baseline_heads have been run. Tests that
    load a champion depend on those files and cannot run without them.
    """
    required = [
        Path("checkpoints/mini_vulcan_pretrained.pt"),
        Path("checkpoints/baseline_heads.pt"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(
            "missing model artifacts: " + ", ".join(missing)
            + " — run: python -m train.pretrain --config configs/tiny.yaml "
              "&& python -m train.train_baseline_heads --config configs/tiny.yaml"
        )
    return required
