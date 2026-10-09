from state import RAGState
from edges import should_continue
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage

class RAGGraph:
    def __init__(self,model,tools):
        graph = StateGraph(RAGState)
        graph.add_node("reason", self.reason_node)
        graph.add_node("tool",self.tool_node)
        graph.add_conditional_edges("reason",should_continue, {True: "tool", False: END})
        graph.add_edge("tool","reason")
        graph.set_entry_point("reason")
        self.graph = graph.compile()
        self.model = model.bind_tools(tools)
        self.tools_by_name = {t.name: t for t in tools}

    def reason_node(self, state:RAGState):
        # print(state)
        messages = state["messages"]
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def tool_node(self, state: RAGState):
        results=[]
        last_msg=state["messages"][-1] if state["messages"] else None
        if last_msg and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                tool = self.tools_by_name.get(tool_call['name'])
                if tool:
                    tool_output=tool.invoke(tool_call["args"])
                    results.append(ToolMessage(content=str(tool_output),tool_call_id=tool_call['id'],name=tool_call['name']))
                else:
                    results.append(ToolMessage(content=f"Error: No tool named {tool_call['name']} and available tools are {', '.join(self.tools_by_name)}"))
        return {"messages": results}