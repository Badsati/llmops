import ast
import logging
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.rag import retrieve_context

logger = logging.getLogger(__name__)

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@tool
async def search_documents(query: str) -> str:
    """Search uploaded documents for information relevant to the query."""
    docs = await retrieve_context(query)
    if not docs:
        return "No relevant documents found in the knowledge base."

    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[{source}, p.{page}]: {doc.page_content[:600]}")
    return "\n\n".join(parts)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression using +, -, *, /, ** operators."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


TOOLS = [search_documents, calculate]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x) + list(y)]


def _build_graph():
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        temperature=0.1,
    ).bind_tools(TOOLS)

    tool_node = ToolNode(TOOLS)

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    def call_model(state: AgentState) -> dict:
        response = llm.invoke(list(state["messages"]))
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_agent(query: str) -> dict:
    agent = _build_graph()
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

    messages = result["messages"]
    final = messages[-1]

    tools_used = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tools_used.extend(tc["name"] for tc in msg.tool_calls)

    return {
        "answer": final.content,
        "tools_used": tools_used,
        "steps": len(messages),
    }
