import os
import getpass

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()  # Load environment variables from .env file

# Define variables globally
llm = OpenAI(model="gpt-3.5-turbo-1106")
nodes = None
vector_index = None
service_context = None

# Check for required environment variables
def check_environment_variables():
    required_variables = ["OPENAI_API_KEY", "ACTIVELOOP_TOKEN"]
    missing_variables = []
    for var in required_variables:
        if not os.getenv(var):
            missing_variables.append(var)
    if missing_variables:
        raise EnvironmentError(
            f"The following environment variables are missing: {', '.join(missing_variables)}"
        )

def build_vector_index(nodes):
    global vector_index, service_context  # Access the global variables
# Create a local Deep Lake VectorStore
    dataset_path = "./data/paul_graham/deep_lake_db"
    vector_store = DeepLakeVectorStore(
        dataset_path=dataset_path, 
        overwrite=True)

    # LLM that will answer questions with the retrieved context
    llm = OpenAI(model="gpt-3.5-turbo-1106")
    # We use OpenAI's embedding model "text-embedding-ada-002"
    embed_model = OpenAIEmbedding()

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm,)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    vector_index = VectorStoreIndex(nodes, service_context=service_context, 
                                    storage_context=storage_context, 
                                    show_progress=True)
    return vector_index

data_url = os.getenv("DATA_URL", "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt")


async def main():
     # Check environment variables before proceeding
    check_environment_variables()
    
# Data Download (only if the file doesn't exist)
    data_path = "data/paul_graham/paul_graham_essay.txt"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        import requests
        response = requests.get(data_url)
        with open(data_path, "w") as f:
            f.write(response.text)

    documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
    nodes = node_parser.get_nodes_from_documents(documents)

    # By default, the node/chunks ids are set to random uuids. To ensure same id's per run, we manually set them.
    for idx, node in enumerate(nodes):
      node.id_ = f"node_{idx}"


    print(f"Number of Documents: {len(documents)}")
    print(f"Number of nodes: {len(nodes)} with the current chunk size of {node_parser.chunk_size}")

    vector_index = build_vector_index(nodes)  # Call the new function

    """With the vector index, we can now build a QueryEngine, which generates answers with the LLM and the retrieved chunks of text."""
    query_engine = vector_index.as_query_engine(similarity_top_k=10)
    response_vector = query_engine.query("What are the main things Paul worked on before college?")
    print(response_vector.response)
    
    import evaluate
    await evaluate.run_evaluations(vector_index, nodes, llm, service_context)  # Pass the needed arguments

if __name__ == "__main__":
  import asyncio
  asyncio.run(main())  # Run the async main function in an event loop
