from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


@dataclass(frozen=True)
class CheckpointerResource:
    checkpointer: BaseCheckpointSaver
    manager: AbstractContextManager[BaseCheckpointSaver] | None = None


def create_checkpointer(database_url: str | None = None) -> CheckpointerResource:
    selected_database_url = database_url or os.getenv("DATABASE_URL", "")
    if not selected_database_url.strip():
        return CheckpointerResource(checkpointer=InMemorySaver())

    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    from langgraph.checkpoint.postgres import PostgresSaver

    manager = PostgresSaver.from_conn_string(selected_database_url)
    checkpointer = manager.__enter__()
    checkpointer.setup()

    return CheckpointerResource(checkpointer=checkpointer, manager=manager)
