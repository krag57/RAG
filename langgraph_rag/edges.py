from state import RAGState

def should_continue(state: RAGState):
    last_msg=state["messages"][-1] if state["messages"] else None
    if hasattr(last_msg,"tool_calls") and last_msg.tool_calls:
        return True
    return False