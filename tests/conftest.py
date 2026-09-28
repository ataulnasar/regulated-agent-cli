from pathlib import Path

import pytest

import app.review


@pytest.fixture(autouse=True)
def isolated_runtime_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests offline and away from the real data/pending directory."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setattr(app.review, "PENDING_DIR", tmp_path / "pending")
