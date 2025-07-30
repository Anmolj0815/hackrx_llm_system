import os
from dotenv import load_dotenv
from pinecone import init, Index, ServerlessSpec # <-- CORRECTED: Index and ServerlessSpec are often back at the top level
# If ServerlessSpec is still not found, check the Pinecone client's GitHub for exact usage.
# Sometimes, for free tier, you just use `init` and the `environment` parameter handles the spec implicitly.

import openai

# Load environment variables from .env file
load_dotenv()

# --- Configuration from Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Assuming PINECONE_ENVIRONMENT now holds the specific region name for ServerlessSpec
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
            # `init` is typically enough, and the `environment` parameter helps
            init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            pinecone_initialized = True
            print(f"Pinecone initialized successfully for environment: {PINECONE_ENVIRONMENT}")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

def get_pinecone_index() -> Index:
    """Connects to an existing Pinecone index or creates it if it doesn't exist."""
    initialize_pinecone_once() # Ensure Pinecone is initialized

    try:
        # After `init`, Pinecone's top-level `pinecone.list_indexes()` is used.
        # The `Index` object is directly from the top-level import now.
        
        # Access the top-level Pinecone client instance for operations like list_indexes
        import pinecone as pinecone_client_instance 
        
        if PINECONE_INDEX_NAME not in pinecone_client_instance.list_indexes():
            print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}...")
            
            # For serverless indexes, 'cloud' and 'region' are required.
            # Assuming 'aws' as the cloud, change if your environment is GCP/Azure.
            # PINECONE_ENVIRONMENT variable should contain the region (e.g., 'us-west-2').
            pinecone_client_instance.create_index( # Call create_index on the client instance
                name=PINECONE_INDEX_NAME,
                dimension=1536, # Dimension for text-embedding-ada-002
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT) # Use the imported ServerlessSpec
            )
            print(f"New Pinecone index '{PINECONE_INDEX_NAME}' created.")
        
        # Return the Index object, also imported from the top level
        return Index(PINECONE_INDEX_NAME) # Access Index class directly
    except Exception as e:
        print(f"Failed to get/create Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        raise

# --- OpenAI Embedding Generation ---
def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using OpenAI's text-embedding-ada-002."""
    if not text.strip(): # Handle empty or whitespace-only text
        return [0.0] * 1536 # Return a zero vector or handle as error

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

# --- Pinecone Data Operations ---
def upsert_chunks_to_pinecone(index: Index, chunks: list[str], document_id: str, namespace: str = None):
    """
    Generates embeddings for text chunks and upserts them to Pinecone.
    Each chunk gets a unique ID and metadata linking it to the document.
    """
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip(): # Skip empty chunks
            continue
        try:
            embedding = get_embedding(chunk)
            vector_id = f"{document_id}_chunk_{i}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": chunk, "document_id": document_id} # Store original text in metadata
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

def query_pinecone(index: Index, query_text: str, top_k: int = 5, namespace: str = None) -> list[str]:
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
            include_metadata=True, # Ensure metadata (including 'text') is returned
            namespace=namespace
        )
        # Return the 'text' content from the metadata of the top_k matches
        return [match.metadata['text'] for match in results.matches if 'text' in match.metadata]
    except ValueError as e: # Catch error from get_embedding
        print(f"Query embedding failed: {e}")
        return []
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        raise

# Optional: Function to delete all vectors from a specific document_id (useful for cleanup/testing)
def delete_document_vectors(index: Index, document_id: str, namespace: str = None):
    """Deletes all vectors associated with a specific document_id."""
    try:
        # Pinecone's delete operation can filter by metadata
        index.delete(filter={"document_id": document_id}, namespace=namespace)
        print(f"Deleted vectors for document_id: {document_id} from Pinecone.")
    except Exception as e:
        print(f"Error deleting vectors for document_id {document_id}: {e}")
        raise
