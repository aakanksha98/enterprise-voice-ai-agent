from typing import Final

from backend.app.agent.state import AgentState, AgentStateUpdate, FAQTopic


FAQ_RESPONSES: Final[dict[FAQTopic, str]] = {
    "greeting": "Hello. How can I help with an appointment or business question today?",
    "assistant_identity": "I'm Aster, an AI reception assistant.",
    "capabilities": (
        "I can help with business questions, appointment booking, cancellation, "
        "rescheduling, or connecting you with a person."
    ),
    "courtesy": "You're welcome. Is there anything else I can help with?",
}


def answer_faq(state: AgentState) -> AgentStateUpdate:
    faq_topic = state.get("faq_topic")
    if faq_topic is None:
        raise ValueError("A FAQ topic is required before answering")

    return {
        "workflow_stage": "faq_answered",
        "draft_response": FAQ_RESPONSES[faq_topic],
    }
