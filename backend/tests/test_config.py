import os
from pathlib import Path

from backend.app.config import load_local_env


def test_load_local_env_sets_missing_values_without_overwriting(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=from-file",
                "DATABASE_URL='postgresql://from-file'",
                "EXISTING_VALUE=from-file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING_VALUE", "from-process")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    load_local_env(env_path)

    assert os.environ["OPENAI_API_KEY"] == "from-file"
    assert os.environ["DATABASE_URL"] == "postgresql://from-file"
    assert os.environ["EXISTING_VALUE"] == "from-process"
