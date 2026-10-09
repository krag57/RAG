class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions,
        prompt_template,
        course="llm-zoomcamp",
        model="gpt-5.6-luna"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
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
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

if __name__ == "__main__":
    from dotenv import load_dotenv
    print(load_dotenv("/Users/ramesh/Documents/RAG/.env"))

    from ingest import load_faq_data, build_index
    # from rag_helper import RAGBase
    from openai import OpenAI

    documents = load_faq_data()
    index = build_index(documents)

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

    assistant = RAGBase(
        index=index,
        llm_client=openai_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
    )

    answer = assistant.rag("I just discovered the course. Can I join now?")
    print(answer)
