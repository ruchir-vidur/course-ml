import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# importing the parsed documents from src/parser.py
from parser import parse_vtt_to_documents

def embed():
    print("1. Parsing VTT files")
    docs = parse_vtt_to_documents("../data")


    print("2. Setting up the embedding model")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.embed_model = embed_model
    Settings.llm = None  # Disable LLM usage

    print("3. Setting up the local Qdrant vector store")
    client = qdrant_client.QdrantClient(path="../qdrant_db")

    vector_store = QdrantVectorStore(client=client, collection_name="vidur_course_embeddings" )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"4. Embedding and indexing {len(docs)} chunks. ")
    index = VectorStoreIndex.from_documents(
        docs, 
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\n✅ Success! All data is vectorized and securely stored in Qdrant.")


if __name__ == "__main__":
    embed()

