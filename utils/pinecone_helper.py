import os
from dotenv import load_dotenv
import pinecone
import openai
from sentence_transformers import SentenceTransformer # <-- NEW IMPORT

# Load environment variables from .env file
load_dotenv()

# --- Configuration from Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Still needed for GPT-4o

# Check for essential environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not set in .env. For ServerlessSpec, this should be a region like 'us-west-2'.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")
if not OPENAI_API_KEY: # Check for OpenAI key for GPT-4o
    raise ValueError("OPENAI_API_KEY not set in .env")

# Initialize OpenAI for GPT-4o (still needed for answer generation)
openai.api_key = OPENAI_API_KEY

# --- Initialize the 1024-dimension Sentence Transformer model globally ---
# This model produces 1024-dimensional embeddings, matching your Pinecone index.
try:
    embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5') # This model has 1024 dimensions
    print("SentenceTransformer model 'BAAI/bge-large-en-v1.5' loaded.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    # Consider a fallback or raise a critical error if model can't be loaded
    embedding_model = None # Set to None if loading fails, handle errors in get_embedding


# --- Pinecone Initialization and Index Management ---
pinecone_initialized = False

def initialize_pinecone_once():
    global pinecone_initialized
    if not pinecone_initialized:
        try:
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            pinecone_initialized = True
            print(f"Pinecone initialized successfully for environment: {PINECONE_ENVIRONMENT}")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

def get_pinecone_index() -> pinecone.Index:
    initialize_pinecone_once()

    try:
        # Check if index exists with the correct dimension
        if PINECONE_INDEX_NAME in pinecone.list_indexes():
            index_info = pinecone.describe_index(PINECONE_INDEX_NAME)
            if index_info.dimension != 1024: # Check if existing index has the correct dimension
                print(f"Warning: Existing index '{PINECONE_INDEX_NAME}' has dimension {index_info.dimension}, expected 1024. "
                      "Please delete and recreate index with correct dimension or adjust model.")
                # You might want to raise an error here if the mismatch is critical.
                # For now, we'll try to proceed but expect upsert errors if not 1024.
        else: # Index does not exist, create it
            print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME} with 1024 dimensions...")
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024, # <-- Fixed to 1024 to match BGE model
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
            print(f"New Pinecone index '{PINECONE_INDEX_NAME}' created.")

        return pinecone.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Failed to get/create Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        raise

# --- Embedding Generation (NOW USING SENTENCE TRANSFORMERS) ---
def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using SentenceTransformer (1024 dimensions)."""
    if not embedding_model:
        raise RuntimeError("Embedding model not loaded. Cannot generate embeddings.")
    if not text.strip():
        return [0.0] * 1024 # Return zero vector of correct dimension

    try:
        # The .encode() method returns a numpy array, convert to list
        embedding_np = embedding_model.encode(text, normalize_embeddings=True) # Normalize for cosine similarity
        return embedding_np.tolist()
    except Exception as e:
        print(f"Error generating embedding with SentenceTransformer: {e}")
        raise ValueError(f"Embedding generation failed: {e}")

# --- Pinecone Data Operations (No changes needed in logic) ---
# The type hints for Index should still be pinecone.Index as before
# ... (rest of the file remains the same from the last full code block you received) ...
def upsert_chunks_to_pinecone(index: pinecone.Index, chunks: list[str], document_id: str, namespace: str = None):
    # ... (unchanged) ...

def query_pinecone(index: pinecone.Index, query_text: str, top_k: int = 5, namespace: str = None) -> list[str]:
    # ... (unchanged) ...

def delete_document_vectors(index: pinecone.Index, document_id: str, namespace: str = None):
    # ... (unchanged) ...
