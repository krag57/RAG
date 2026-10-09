import json
from langchain.tools import tool
from weaviate.classes.query import MetadataQuery

def make_weaviate_search_tool(weaviate_client):
    cls = weaviate_client.collections.use("FAQDocuments")
    @tool
    def weaviate_search_tool(query: str):
        """Search the weaviate database for the information."""
        response = cls.query.near_text(
            query=query,
            limit=3,
            return_metadata=MetadataQuery(distance=True)
        )
        results_list = [obj.properties for obj in response.objects]
        return json.dumps(results_list, indent=2)
    return weaviate_search_tool