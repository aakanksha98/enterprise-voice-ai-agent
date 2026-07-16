import pytest

from backend.app.agent.graph import agent_graph
from backend.app.agent.routing import route_validated_input


def test_valid_message_routes_to_ready_for_planning() -> None:
    result = agent_graph.invoke({"user_message": "  Book an appointment  "})

    assert result == {
        "user_message": "  Book an appointment  ",
        "normalized_message": "Book an appointment",
        "input_status": "valid",
        "workflow_stage": "ready_for_planning",
    }


def test_empty_message_routes_to_rejection() -> None:
    result = agent_graph.invoke({"user_message": "   "})

    assert result == {
        "user_message": "   ",
        "normalized_message": "",
        "input_status": "invalid",
        "workflow_stage": "rejected",
        "validation_error": "user_message must not be empty",
    }


@pytest.mark.parametrize(
    ("user_message", "expected_nodes"),
    [
        ("What services do you offer?", ["validate_input", "ready_for_planning"]),
        ("\t", ["validate_input", "reject_invalid_input"]),
    ],
)
def test_conditional_edge_selects_one_terminal_branch(
    user_message: str,
    expected_nodes: list[str],
) -> None:
    updates = agent_graph.stream(
        {"user_message": user_message},
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == expected_nodes


def test_router_rejects_unvalidated_state() -> None:
    with pytest.raises(ValueError, match="must be validated"):
        route_validated_input({"user_message": "Book an appointment"})
