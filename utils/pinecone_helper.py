import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
from sentence_transformers import SentenceTransformer
from typing import Any

# Load environment variables from .env file
load_dotenv()

# --- Configuration from Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., 'us-east-1'
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for essential environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not set in .env. For ServerlessSpec, this should be a region like 'us-east-1'.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# Initialize OpenAI for GPT-4o
openai.api_key = OPENAI_API_KEY

# --- Initialize the 1024-dimension Sentence Transformer model globally ---
embedding_model = None
try:
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load SentenceTransformer model: {e}")
    embedding_model = None

# --- Pinecone Client Instance (GLOBAL) ---
pc = None

def initialize_pinecone_client():
    """Initializes the global Pinecone client instance."""
    global pc
    if pc is None:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            print("Pinecone client instance created.")
        except Exception as e:
            print(f"Error creating Pinecone client instance: {e}")
            raise

# --- Corrected get_pinecone_index function ---
def get_pinecone_index() -> Any:
    """Connects to an existing Pinecone index or creates it if it doesn't exist."""
    initialize_pinecone_client()

    if pc is None:
        raise RuntimeError("Pinecone client instance is not initialized.")

    try:
        # Check if index exists by listing current indexes.
        # --- CRITICAL CHANGE: Added () to call the .names() method ---
        existing_indexes_names = pc.list_indexes().names() 
        
        if PINECONE_INDEX_NAME not in existing_indexes_names:
            print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
            print(f"New Pinecone index '{PINECONE_INDEX_NAME}' created.")
        else:
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")
        
        # Return the Index object via the client instance
        return pc.Index(PINECONE_INDEX_NAME) 
    except Exception as e:
        print(f"Failed to get/create Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        raise

# --- Embedding Generation ---
def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using SentenceTransformer (BAAI/bge-large-en-v1.5)."""
    if embedding_model is None:
        raise RuntimeError("Embedding model is not initialized. Cannot generate embeddings.")
    if not text.strip():
        return [0.0] * 384

    try:
        embedding_np = embedding_model.encode(text, normalize_embeddings=True)
        return embedding_np.tolist()
    except Exception as e:
        print(f"Error generating embedding with SentenceTransformer: {e}")
        raise ValueError(f"Embedding generation failed: {e}")

# --- Pinecone Data Operations ---
def upsert_chunks_to_pinecone(index: Any, chunks: list[str], document_id: str, namespace: str = None):
    """
    Generates embeddings for text chunks and upserts them to Pinecone.
    Each chunk gets a unique ID and metadata linking it to the document.
    """
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            embedding = get_embedding(chunk)
            vector_id = f"{document_id}_chunk_{i}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": chunk, "document_id": document_id}
            })
        except ValueError as e:
            print(f"Skipping chunk {i} due to embedding error: {e}")
            continue

    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            print(f"Upserted {len(vectors_to_upsert)} chunks for document {document_id} to Pinecone index '{index.name}'.")
        except Exception as e:
            print(f"Error upserting vectors to Pinecone: {e}")
            raise
    else:
        print(f"No valid chunks to upsert for document {document_id}.")

def query_pinecone(index: Any, query_text: str, top_k: int = 5, namespace: str = None) -> list[str]:
    """
    Queries Pinecone with an embedding of the query text and returns
    the most relevant chunks' text content.
    """
    if not query_text.strip():
        return []

    try:
        query_embedding = get_embedding(query_text)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return [match.metadata['text'] for match in results.matches if 'text' in match.metadata]
    except ValueError as e:
        print(f"Query embedding failed: {e}")
        return []
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        raise

def delete_document_vectors(index: Any, document_id: str, namespace: str = None):
    """Deletes all vectors associated with a specific document_id."""
    try:
        index.delete(filter={"document_id": document_id}, namespace=namespace)
        print(f"Deleted vectors for document_id: {document_id} from Pinecone.")
    except Exception as e:
        print(f"Error deleting vectors for document_id {document_id}: {e}")
        raise
