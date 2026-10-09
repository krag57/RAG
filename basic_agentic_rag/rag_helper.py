import weaviate
from weaviate.classes.query import MetadataQuery
import argparse
import os
import json

class RAGBase:
    def __init__(
        self,
        weviate_client,
        llm_client,
        instructions,
        weviate_class='FAQDocuments',
        course="llm-zoomcamp",
        model="gpt-5.6-luna",
        verbosity=False
    ):
        self.weviate_client = weviate_client
        self.llm_client = llm_client
        self.weaviate_class = weviate_class
        self.instructions = instructions
        self.course = course
        self.model = model
        self.verbosity=verbosity

        self.search_tool = {
            "type": "function",
            "name": "search",
            "description": "Search the FAQ database for entries matching the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text to look up in the course FAQ."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }

    def search(self, query, num_results=2):
        cls = self.weviate_client.collections.use("FAQDocuments")
        response = cls.query.near_text(
            query=query,
            limit=num_results,
            return_metadata=MetadataQuery(distance=True)
        )
        results_list = [obj.properties for obj in response.objects]
        return results_list

    def make_call(self, call):
        args = json.loads(call.arguments)

        if call.name == "search":
            result = self.search(**args)
        # print(result)
        result_json = json.dumps(result, indent=2)

        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": result_json,
        }
    
    def agent_loop(self, question) -> str:
        messages = [
                    {"role": "developer", "content": self.instructions},
                    {"role": "user", "content": question}
                ]

        it = 1

        while True:
            if self.verbosity: print(f"iteration #{it}...")
            has_function_calls = False

            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                tools=[self.search_tool]
            )

            messages.extend(response.output)

            for item in response.output:
                if item.type == "function_call":
                    if self.verbosity: print("function_call:", item.name, item.arguments)
                    call_output = self.make_call(item)
                    messages.append(call_output)
                    has_function_calls = True

                elif item.type == "message":
                    print("ASSISTANT:")
                    last_answer = item.content[0].text
                    messages.append({'role':'assistant', 'content':last_answer})
                    print(last_answer)

            it = it + 1
            if has_function_calls == False:
                break

        return last_answer

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    weaviate_url = os.environ["WEAVIATE_URL"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]   

    # from ingest import load_faq_data, build_index
    # from rag_helper import RAGBase
    from openai import OpenAI
    openai_client = OpenAI()

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
        assistant = RAGBase(
            weviate_client=client,
            weviate_class='FAQDocuments',
            llm_client=openai_client,
            instructions=instructions,
            verbosity=False
        )

        answer = assistant.agent_loop(args.question)
        print(answer)
    
