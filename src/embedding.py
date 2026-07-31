from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# importing the parsed documents from src/parser.py
from parser import parse_vtt_to_documents
from qdrant_connection import create_qdrant_client, qdrant_mode_label

def embed():
    print("1. Parsing VTT files")
    docs = parse_vtt_to_documents("../data")


    print("2. Setting up the embedding model")
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.embed_model = embed_model
    Settings.llm = None  # Disable LLM usage

    print("3. Setting up the Qdrant vector store")
    client = create_qdrant_client()
    print(f"   Using {qdrant_mode_label()} Qdrant backend.")

    vector_store = QdrantVectorStore(client=client, collection_name="vidur_course_embeddings" )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"4. Embedding and indexing {len(docs)} chunks. ")
    index = VectorStoreIndex.from_documents(
        docs, 
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\n✅ Success! All data is vectorized and stored in Qdrant.")


if __name__ == "__main__":
    embed()

