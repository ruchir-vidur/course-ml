import os
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.groq import Groq
# NEW: Import filtering tools
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from qdrant_connection import create_qdrant_client, qdrant_mode_label

# Load the Groq API key from course-ml/.env (nothing secret is hardcoded).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except Exception:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to course-ml/.env (see .env.example).")

def setup_query_engine(selected_course: str):
    print("1. Loading Embedding Model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("2. Connecting to Llama 3 via Groq...")
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    
    print("3. Connecting to Qdrant Database...")
    client = create_qdrant_client()
    print(f"   Using {qdrant_mode_label()} Qdrant backend.")
    vector_store = QdrantVectorStore(client=client, collection_name="vidur_course_embeddings")
    
    print("4. Loading Index...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # --- NEW: Filter by Course ---
    # This tells Qdrant to ONLY retrieve vectors where course_id matches your selection
    course_filter = MetadataFilters(
        filters=[ExactMatchFilter(key="course_id", value=selected_course)]
    )
    
    # --- Prompt Engineering ---
    qa_prompt_tmpl_str = (
        "You are an expert AI teaching assistant for a course. \n"
        "Below is context from the course transcripts. Each piece of context has metadata attached.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the user's query.\n"
        "You MUST explicitly cite the exact session and provide the video link at the end of your answer.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    
    # Pass the filter into the query engine
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=course_filter,
        text_qa_template=qa_prompt_tmpl
    )
    
    return query_engine

if __name__ == "__main__":
    print("=== Welcome to the Course AI ===")
    # Ask the user which course they want to talk to before loading the engine
    course_name = input("Which course do you want to chat with? (e.g., 'Aligning Mind Space & Energy'): ")
    
    engine = setup_query_engine(course_name)
    print(f"\n✅ AI is ready and locked to course: {course_name}! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("AI is thinking...")
        response = engine.query(user_input)
        
        print(f"\nCourse AI:\n{response}")
        
        print("\n" + "="*50)
        print("DEBUG: Source Context Used by AI")
        print("="*50)
        for i, node in enumerate(response.source_nodes):
            metadata = node.metadata
            url_with_timestamp = f"{metadata.get('video_url')}?t={metadata.get('start_time_seconds')}"
            print(f"Source {i+1}: Session {metadata.get('session')} | Link: {url_with_timestamp}")
            print(f"Text Snippet: {node.text[:100]}...\n")