import os
from dotenv import load_dotenv
from openai import OpenAI
import weaviate
import argparse
from langchain_openai import ChatOpenAI
from graph import RAGGraph
from state import RAGState
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from tools import make_weaviate_search_tool


load_dotenv(".env")

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]   

# openai_client = OpenAI()


instructions = """
    You're a course teaching assistant.
    You're given a question from a course student and your task is to answer it.

    If you want to look up information, use the search function. 
    Use as many keywords from the user question as possible when making first requests.

    Make multiple searches. First perform search, analyze the results 
    and then perform more searches. 
    """.strip()

parser = argparse.ArgumentParser(description="RAG Application on QA for the course")

parser.add_argument("-q","--question", type=str, help="the question you have about the course")

args = parser.parse_args()




with weaviate.connect_to_weaviate_cloud(cluster_url=weaviate_url,auth_credentials=weaviate_api_key) as client:
    model = ChatOpenAI(
                        model="gpt-5.6-luna", 
                        temperature=0,
                        reasoning_effort="none" 
                    )
    tools=[make_weaviate_search_tool(weaviate_client=client)]
    # response=tools.invoke("can I register now")
    # print(response)
    # 
    RAG_graph=RAGGraph(model,tools)

    initial_state: RAGState = {
        "messages":[SystemMessage(content=instructions),
                    HumanMessage(content="can I register now")]
    }

    # print(initial_state)
    

    final_state=RAG_graph.graph.invoke(initial_state, config={"recursion_limit":10})
    # for message in final_state['messages']:
    #     print(message)
    final_message=final_state['messages'][-1]
    print(final_message)

