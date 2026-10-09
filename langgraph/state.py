import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

class RAGState(TypedDict):
    """State of the RAG"""
    messages: Annotated[list[BaseMessage],operator.add]