import os
from dotenv import load_dotenv
import pinecone # Import the whole module
import openai
from sentence_transformers import SentenceTransformer # NEW IMPORT for embedding model

# Load environment variables from .env file
load_dotenv()

# --- Configuration from Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Assuming PINECONE_ENVIRONMENT now holds the specific region name for ServerlessSpec (e.g., 'us-east-1')
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for essential environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not set in .env. For ServerlessSpec, this should be a region like 'us-east-1'.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")
if not OPENAI_API_KEY: # Check for OpenAI key for GPT-4o
    raise ValueError("OPENAI_API_KEY not set in .env")

# Initialize OpenAI for GPT-4o (still needed for answer generation)
openai.api_key = OPENAI_API_KEY

# --- Initialize the 1024-dimension Sentence Transformer model globally ---
# This model produces 1024-dimensional embeddings, matching your existing Pinecone index.
try:
    print("Loading SentenceTransformer model 'BAAI/bge-large-en-v1.5'...")
    # This model will be downloaded and loaded once when the application starts.
    # It has 1024 dimensions.
    embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    print("SentenceTransformer model 'BAAI/bge-large-en-v1.5' loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load SentenceTransformer model: {e}")
    # In a production app, you might want to raise a critical error or have a fallback.
    embedding_model = None # Set to None if loading fails, handle errors in get_embedding


# --- Pinecone Initialization and Index Management ---
pinecone_initialized = False

def initialize_pinecone_once():
    """Initializes Pinecone globally, only once."""
    global pinecone_initialized
    if not pinecone_initialized:
        try:
            # Use pinecone.init() directly
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            pinecone_initialized = True
            print(f"Pinecone initialized successfully for environment: {PINECONE_ENVIRONMENT}")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

def get_pinecone_index() -> pinecone.Index:
    """Connects to an existing Pinecone index or creates it if it doesn't exist."""
    initialize_pinecone_once() # Ensure Pinecone is initialized

    try:
        # Check if index exists with the correct dimension for BGE model (1024)
        if PINECONE_INDEX_NAME in pinecone.list_indexes():
            index_info = pinecone.describe_index(PINECONE_INDEX_NAME)
            if index_info.dimension != 1024:
                print(f"WARNING: Existing Pinecone index '{PINECONE_INDEX_NAME}' has dimension {index_info.dimension}, but expected 1024 (for BGE model). "
                      "This will cause upsert errors unless corrected. Consider deleting and recreating the index if you control it.")
                # We will proceed, but expect upsert errors if the dimension truly mismatches at runtime.
                # For a contest, you might want to crash here if index dim is wrong to avoid silent failures.
        else: # Index does not exist, create it
            print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME} with 1024 dimensions...")
            # For serverless indexes, 'cloud' and 'region' are required.
            # Assuming 'aws' as the cloud, as it's common for 'us-east-1'. Change if your environment is GCP/Azure.
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024, # <-- Set dimension to 1024 to match BGE model
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
            print(f"New Pinecone index '{PINECONE_INDEX_NAME}' created.")
        
        # Return the Index object, accessed as an attribute of the imported pinecone module
        return pinecone.Index(PINECONE_INDEX_NAME) 
    except Exception as e:
        print(f"Failed to get/create Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        raise

# --- Embedding Generation (NOW USING SENTENCE TRANSFORMERS) ---
def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using SentenceTransformer (BAAI/bge-large-en-v1.5)."""
    if embedding_model is None: # Check if model loaded successfully at startup
        raise RuntimeError("Embedding model is not initialized. Cannot generate embeddings.")
    if not text.strip():
        # Return a zero vector of the correct dimension (1024 for BGE-Large)
        return [0.0] * 1024

    try:
        # The .encode() method returns a numpy array, convert to list.
        # normalize_embeddings=True is important for cosine similarity.
        embedding_np = embedding_model.encode(text, normalize_embeddings=True)
        return embedding_np.tolist()
    except Exception as e:
        print(f"Error generating embedding with SentenceTransformer: {e}")
        raise ValueError(f"Embedding generation failed: {e}")

# --- Pinecone Data Operations (Type hints updated to pinecone.Index) ---
def upsert_chunks_to_pinecone(index: pinecone.Index, chunks: list[str], document_id: str, namespace: str = None):
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

def query_pinecone(index: pinecone.Index, query_text: str, top_k: int = 5, namespace: str = None) -> list[str]:
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
        return [match.metadata['text'] for match in results.matches if 'text' in match.metadata]
    except ValueError as e: # Catch error from get_embedding
        print(f"Query embedding failed: {e}")
        return []
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        raise

def delete_document_vectors(index: pinecone.Index, document_id: str, namespace: str = None):
    """Deletes all vectors associated with a specific document_id."""
    try:
        index.delete(filter={"document_id": document_id}, namespace=namespace)
        print(f"Deleted vectors for document_id: {document_id} from Pinecone.")
    except Exception as e:
        print(f"Error deleting vectors for document_id {document_id}: {e}")
        raise
