import os
from dotenv import load_dotenv
import asyncio
import uuid
import pinecone # Import pinecone directly here for initialization check

# Import functions from our utils package
from utils.document_parser import parse_document, chunk_text
from utils.pinecone_helper import get_pinecone_index, upsert_chunks_to_pinecone #, delete_document_vectors

load_dotenv()

# --- IMPORTANT: Replace these with your actual public URLs for static policy documents ---
# These documents will be ingested into your Pinecone index.
# Make sure these URLs are publicly accessible (e.g., from Azure Blob Storage, AWS S3, etc.)
STATIC_DOCUMENT_URLS = [
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    # Add your other policy document URLs here:
    # "https://your-storage-account.blob.core.windows.net/your-container/policy2.pdf",
    # "https://your-storage-account.blob.core.windows.net/your-container/contract_terms.docx", # Example for .docx if you implement
    # "https://your-storage-account.blob.core.windows.net/your-container/email_policy.eml",   # Example for .eml if you implement
]

# Using a consistent namespace for static documents can be helpful for management
STATIC_DOCS_NAMESPACE = "static-policies" 

async def ingest_static_documents():
    print("Starting static document ingestion...")
    pinecone_index = None
    try:
        pinecone_index = get_pinecone_index()
    except Exception as e:
        print(f"Could not initialize Pinecone index. Aborting ingestion: {e}")
        return

    for doc_url in STATIC_DOCUMENT_URLS:
        document_local_id = str(uuid.uuid4()) # A unique ID for this ingestion run/document
        try:
            print(f"Processing document for ingestion: {doc_url}")
            document_text = await asyncio.to_thread(parse_document, doc_url) # Await for async parse_document
            
            if not document_text.strip():
                print(f"Skipping {doc_url}: No text extracted.")
                continue

            chunks = chunk_text(document_text, chunk_size=1000, chunk_overlap=200)
            if not chunks:
                print(f"Skipping {doc_url}: No chunks generated.")
                continue

            await asyncio.to_thread(
                upsert_chunks_to_pinecone, 
                pinecone_index, 
                chunks, 
                document_local_id, 
                namespace=STATIC_DOCS_NAMESPACE
            )
            print(f"Successfully ingested {doc_url} with ID {document_local_id} into namespace '{STATIC_DOCS_NAMESPACE}'.")
        except Exception as e:
            print(f"Failed to ingest {doc_url}: {e}")
            # Continue with other documents even if one fails

    print("Static document ingestion complete.")

if __name__ == "__main__":
    # Ensure all necessary environment variables are set before running
    missing_vars = []
    if not os.getenv("PINECONE_API_KEY"): missing_vars.append("PINECONE_API_KEY")
    if not os.getenv("PINECONE_ENVIRONMENT"): missing_vars.append("PINECONE_ENVIRONMENT")
    if not os.getenv("PINECONE_INDEX_NAME"): missing_vars.append("PINECONE_INDEX_NAME")
    if not os.getenv("OPENAI_API_KEY"): missing_vars.append("OPENAI_API_KEY")

    if missing_vars:
        print(f"Error: Missing environment variables for ingestion: {', '.join(missing_vars)}. Please set them in .env.")
    else:
        asyncio.run(ingest_static_documents())
