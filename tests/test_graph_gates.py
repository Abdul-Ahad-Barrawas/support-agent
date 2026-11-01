from src.agent.graph import route_after_priority


def test_route_after_priority_escalates_critical():
    state = {
        "should_escalate": True
    }

    output = route_after_priority(state)

    assert output == "escalate"


def test_route_after_priority_retrieves_otherwise():
    state = {
        "should_escalate": False
    }

    output = route_after_priority(state)

    assert output == "retrieve"
