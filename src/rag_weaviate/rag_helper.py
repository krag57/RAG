import weaviate
from weaviate.classes.query import MetadataQuery
import argparse
import os

class RAGBase:
    def __init__(
        self,
        weviate_client,
        llm_client,
        instructions,
        prompt_template,
        weviate_class='FAQDocuments',
        course="llm-zoomcamp",
        model="gpt-5.6-luna"
    ):
        self.weviate_client = weviate_client
        self.llm_client = llm_client
        self.weaviate_class = weviate_class
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        cls = self.weviate_client.collections.use("FAQDocuments")
        response = cls.query.near_text(
            query=query,
            limit=num_results,
            return_metadata=MetadataQuery(distance=True)
        )
        return response

    def build_context(self, search_results):
        lines = []

        for doc in search_results.objects:
            lines.append(doc.properties["section"])
            lines.append("Q: " + doc.properties["question"])
            lines.append("A: " + doc.properties['answer'])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        # print(search_results)
        prompt = self.build_prompt(query, search_results)
        # print(prompt)
        answer = self.llm(prompt)
        return answer

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    weaviate_url = os.environ["WEAVIATE_URL"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]   

    # from ingest import load_faq_data, build_index
    # from rag_helper import RAGBase
    from openai import OpenAI
    openai_client = OpenAI()



    INSTRUCTIONS = """
    Your task is to answer questions from the course participants
    based on the provided context.

    Use the context to find relevant information and provide accurate
    answers. If the answer is not found in the context,
    respond with "I don't know."
    """

    PROMPT_TEMPLATE = """
    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    parser = argparse.ArgumentParser(
        description="RAG Application on QA for the course"
    )

    parser.add_argument("-q","--question", type=str, help="the question you have about the course")

    args = parser.parse_args()

    with weaviate.connect_to_weaviate_cloud(cluster_url=weaviate_url,auth_credentials=weaviate_api_key) as client:
        assistant = RAGBase(
            weviate_client=client,
            weviate_class='FAQDocuments',
            llm_client=openai_client,
            instructions=INSTRUCTIONS,
            prompt_template=PROMPT_TEMPLATE,
        )

        answer = assistant.rag(args.question)
        print(answer)
    
