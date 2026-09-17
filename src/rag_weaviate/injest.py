import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery
import weaviate.classes.config as PropertyConfig
import requests
import pickle

load_dotenv()

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]


docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()

documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"""{url_prefix}{course["path"]}"""

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()
    documents.extend(course_data)

drop_keys = {'id'}

# Process each dictionary in the list
documents = [{k: v for k, v in d.items() if k not in drop_keys} for d in documents]


schema_properties = [
    # Include in vector
    PropertyConfig.Property(name="question", data_type=PropertyConfig.DataType.TEXT),
    PropertyConfig.Property(name="answer", data_type=PropertyConfig.DataType.TEXT),
    PropertyConfig.Property(name="course", data_type=PropertyConfig.DataType.TEXT,skip_vectorization=True),
    PropertyConfig.Property(name="section", data_type=PropertyConfig.DataType.TEXT, skip_vectorization=True)
]

# Step 1.1: Connect to your Weaviate Cloud instance
with weaviate.connect_to_weaviate_cloud(cluster_url=weaviate_url,auth_credentials=weaviate_api_key) as client:

    # Step 1.2: Create a collection
    client.collections.delete_all()

    faq_docs = client.collections.create(
        name="FAQDocuments",
        properties=schema_properties,
        vector_config=Configure.Vectors.text2vec_weaviate(),  # Configure the Weaviate Embeddings vectorizer
    )

    faq_docs = client.collections.use("FAQDocuments")
    faq_docs.data.ingest(documents)

    print(f"Imported & vectorized {len(faq_docs)} objects into the FAQDocuments collection")

# Fetch the collection from Weaviate
with weaviate.connect_to_weaviate_cloud(cluster_url=weaviate_url,auth_credentials=weaviate_api_key) as client:
    collection = client.collections.get("FAQDocuments")

    # Inspect the live configuration
    config = collection.config.get()

    for prop in config.properties:
        print(f"Property Name: {prop.name} | Data Type: {prop.data_type}")


#retrieve docs
with weaviate.connect_to_weaviate_cloud(cluster_url=weaviate_url,auth_credentials=weaviate_api_key) as client:
    cls=client.collections.use("FAQDocuments")
    response = cls.query.near_text(
        query="I just discovered the course. Can I join now?",
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )

for o in response.objects:
    # print(o)
    print(o.properties['answer'])
    print(o.metadata.distance)



