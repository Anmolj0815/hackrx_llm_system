import os
from dotenv import load_dotenv
import pinecone # <-- Import the whole module
import openai

# Load environment variables from .env file
load_dotenv()

# --- Configuration from Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for essential environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not set in .env. For ServerlessSpec, this should be a region like 'us-west-2'.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# Initialize OpenAI for embeddings
openai.api_key = OPENAI_API_KEY

# --- Pinecone Initialization and Index Management ---
pinecone_initialized = False

def initialize_pinecone_once():
    """Initializes Pinecone globally, only once."""
    global pinecone_initialized
    if not pinecone_initialized:
        try:
            # Use pinecone.init()
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            pinecone_initialized = True
            print(f"Pinecone initialized successfully for environment: {PINECONE_ENVIRONMENT}")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

def get_pinecone_index() -> pinecone.Index: # Type hint also uses pinecone.Index
    """Connects to an existing Pinecone index or creates it if it doesn't exist."""
    initialize_pinecone_once() # Ensure Pinecone is initialized

    try:
        if PINECONE_INDEX_NAME not in pinecone.list_indexes():
            print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}...")
            
            # Use pinecone.ServerlessSpec or pinecone.PodSpec as needed.
            # ServerlessSpec is common for new free tiers.
            # Assuming 'aws' as the cloud, change if your environment is GCP/Azure.
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536, # Dimension for text-embedding-ada-002
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
            print(f"New Pinecone index '{PINECONE_INDEX_NAME}' created.")
        
        # Access the Index class directly as an attribute of the imported pinecone module
        return pinecone.Index(PINECONE_INDEX_NAME) 
    except Exception as e:
        print(f"Failed to get/create Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        raise

# --- OpenAI Embedding Generation (No changes needed) ---
def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using OpenAI's text-embedding-ada-002."""
    if not text.strip():
        return [0.0] * 1536

    try:
        response = openai.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except openai.APIError as e:
        print(f"OpenAI Embedding API Error: {e}")
        raise ValueError(f"Failed to get embedding: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}")
        raise ValueError(f"Embedding generation failed: {e}")

# --- Pinecone Data Operations (No changes needed in logic, just type hints if they use `Index` explicitly) ---
def upsert_chunks_to_pinecone(index: pinecone.Index, chunks: list[str], document_id: str, namespace: str = None): # Type hint uses pinecone.Index
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

def query_pinecone(index: pinecone.Index, query_text: str, top_k: int = 5, namespace: str = None) -> list[str]: # Type hint uses pinecone.Index
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

def delete_document_vectors(index: pinecone.Index, document_id: str, namespace: str = None): # Type hint uses pinecone.Index
    """Deletes all vectors associated with a specific document_id."""
    try:
        index.delete(filter={"document_id": document_id}, namespace=namespace)
        print(f"Deleted vectors for document_id: {document_id} from Pinecone.")
    except Exception as e:
        print(f"Error deleting vectors for document_id {document_id}: {e}")
        raise
