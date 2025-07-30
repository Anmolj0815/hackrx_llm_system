import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import openai
import asyncio # For running sync Pinecone operations in a non-blocking way

# Import functions from our utils package
from utils.document_parser import parse_document, chunk_text
# Ensure this import matches the updated pinecone_helper.py
from utils.pinecone_helper import get_pinecone_index, upsert_chunks_to_pinecone, query_pinecone # , delete_document_vectors

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# It's important that PINECONE_ENVIRONMENT is set to your actual region (e.g., 'us-east-1')
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") 


# Ensure all critical environment variables are set at startup
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")
if not API_AUTH_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN environment variable not set. Please set a secret token for authentication.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME environment variable not set for Pinecone index.")
if not PINECONE_ENVIRONMENT:
    raise RuntimeError("PINECONE_ENVIRONMENT environment variable not set (e.g., 'us-east-1').")


# Initialize OpenAI API client
openai.api_key = OPENAI_API_KEY

# --- FastAPI App Setup ---
app = FastAPI(
    title="HackRx LLM Document Q&A System",
    description="Processes natural language queries against unstructured documents using LLMs and Vector DB. Optimized for accuracy, token efficiency, latency, reusability, and explainability.",
    version="1.0.0"
)

# --- Global Pinecone Index Instance ---
# This will be initialized once when the app starts
pinecone_index = None
STATIC_DOCS_NAMESPACE = "static-policies" # Namespace for pre-ingested documents

@app.on_event("startup")
async def startup_event():
    """Initializes Pinecone index when the FastAPI application starts up."""
    global pinecone_index
    print("Attempting to initialize Pinecone index on startup...")
    try:
        pinecone_index = get_pinecone_index()
        print("Pinecone index initialized successfully and ready for use.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Pinecone index on startup: {e}")
        # In a production system, you might want to raise an error here to prevent startup
        # For a contest, we'll allow startup but API calls will fail gracefully if index isn't ready.

# --- Pydantic Models for Request/Response ---
class HackRxRequest(BaseModel):
    documents: str # Assuming one document URL per request, as per sample
    questions: list[str]

class HackRxResponse(BaseModel):
    answers: list[str]

# --- Authentication Dependency ---
async def verify_api_key(request: Request):
    """Authenticates requests using a Bearer token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    
    token_prefix = "Bearer "
    if not auth_header.startswith(token_prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format. Must be 'Bearer <token>'")
    
    token = auth_header[len(token_prefix):].strip()
    if token != API_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return token

# --- RAG Logic ---
async def get_answer_from_llm(question: str, context: str) -> str:
    """
    Uses GPT-4o to generate an answer based on the question and provided context.
    The answer should include decision, amount, and justification with clause references.
    """
    if not context.strip():
        # Fallback if no relevant context is found
        return f"Decision: Cannot Determine. Justification: No relevant information found in the provided documents to answer: '{question}'. Please provide more context."

    system_prompt = (
        "You are an expert assistant for processing policy documents, contracts, and emails. "
        "Your task is to answer user questions based *strictly and exclusively* on the 'Context' provided. "
        "For insurance-related queries, you must clearly state a 'Decision' (e.g., 'Approved', 'Rejected', 'Covered', 'Not Covered', 'Conditional Approval') "
        "and, if applicable, an 'Amount' (e.g., '$5000', 'Full Sum Insured'). "
        "Always provide a 'Justification' by quoting the *exact relevant sentence(s) or clause(s) directly from the provided Context*. "
        "If the Context does not contain enough information to make a definitive decision or answer, state 'Decision: Cannot Determine' and explain why. "
        "Do not invent information or use external knowledge. Be concise and precise. "
        "Format your answer as a clear, single paragraph or a series of bullet points if multiple clauses are needed."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        response = await openai.chat.completions.create( # Use await for async API call
            model="gpt-4o",  # Using GPT-4o as the powerful reasoning model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # Low temperature for factual, less creative answers
            max_tokens=700, # Increased max_tokens to allow for detailed justifications
            timeout=25 # Set a timeout for the LLM call to fit within 30s overall API timeout
        )
        return response.choices[0].message.content.strip()
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error (HTTP {e.status_code}): {e.response}")
        if e.status_code == 429: # Rate limit
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="OpenAI API rate limit exceeded. Please try again shortly.")
        elif e.status_code == 401: # Invalid API key
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid OpenAI API Key configured on server.")
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OpenAI API error: {e.status_code}")
    except openai.APITimeoutError:
        print("OpenAI API call timed out.")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="OpenAI API call timed out.")
    except Exception as e:
        print(f"An unexpected error occurred during LLM generation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal server error occurred during answer generation: {e}")

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx_query(
    request: HackRxRequest,
    api_key_verified: str = Depends(verify_api_key) # Authentication applied here
):
    """
    Processes natural language queries against provided unstructured documents.
    Downloads documents, extracts text, performs semantic search, and generates answers using LLMs.
    """
    # Ensure Pinecone index is initialized; if not, return service unavailable
    if pinecone_index is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pinecone index not initialized. Service is unavailable.")

    document_url = request.documents
    questions = request.questions
    all_answers = []
    
    current_request_doc_id = str(uuid.uuid4())
    dynamic_doc_namespace = f"dynamic-doc-{current_request_doc_id}" # Unique namespace for this request's dynamic doc

    try:
        # 1. Document Ingestion & Processing (for the document provided in the current request)
        print(f"Processing dynamic document from URL: {document_url}")
        document_text = await asyncio.to_thread(parse_document, document_url)
        
        if not document_text.strip():
            print(f"No text extracted from document {document_url}. Answers might be 'Cannot Determine'.")
        else:
            chunks = chunk_text(document_text, chunk_size=1000, chunk_overlap=200)
            if chunks:
                await asyncio.to_thread(
                    upsert_chunks_to_pinecone, 
                    pinecone_index, 
                    chunks, 
                    current_request_doc_id, 
                    namespace=dynamic_doc_namespace
                )
            else:
                print(f"No valid chunks generated for document {document_url}.")

        # 2. Process each question using RAG (searching across static and dynamic documents)
        for i, question in enumerate(questions):
            print(f"Answering question {i+1}/{len(questions)}: {question}")
            
            # Query the static documents namespace
            static_relevant_chunks = await asyncio.to_thread(
                query_pinecone, 
                pinecone_index, 
                question, 
                top_k=3, 
                namespace=STATIC_DOCS_NAMESPACE
            )
            
            # Query the dynamic document's namespace (if it had text and chunks)
            dynamic_relevant_chunks = []
            if document_text.strip() and chunks: # Only query if dynamic doc was successfully processed
                 dynamic_relevant_chunks = await asyncio.to_thread(
                    query_pinecone, 
                    pinecone_index, 
                    question, 
                    top_k=2, # Get fewer from dynamic as it's just one doc
                    namespace=dynamic_doc_namespace
                )
            
            # Combine and de-duplicate chunks (simple join for now)
            combined_chunks = static_relevant_chunks + dynamic_relevant_chunks
            combined_chunks = list(dict.fromkeys(combined_chunks)) # Remove duplicates, preserve order

            context = "\n".join(combined_chunks)
            if not context.strip():
                print(f"No relevant context found for question: {question}")
                answer = f"Decision: Cannot Determine. Justification: No relevant information found in the provided documents or static knowledge base to answer: '{question}'."
            else:
                answer = await get_answer_from_llm(question, context)
            
            all_answers.append(answer)

    except HTTPException as e:
        raise e
    except ValueError as e:
        print(f"Client-side error processing request: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print(f"An unhandled error occurred during request processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected internal error occurred: {e}")
    finally:
        # Optional: Clean up dynamic document vectors after processing request.
        # This is important for "clean state" between requests in a contest environment
        # or to avoid accumulating data if documents are truly ephemeral.
        try:
            if dynamic_doc_namespace and pinecone_index:
                await asyncio.to_thread(
                    pinecone_index.delete, 
                    delete_all=True, 
                    namespace=dynamic_doc_namespace
                )
                print(f"Cleaned up dynamic document namespace: {dynamic_doc_namespace}")
        except Exception as e:
            print(f"Error during dynamic document cleanup for {dynamic_doc_namespace}: {e}")

    return HackRxResponse(answers=all_answers)

# --- Health Check (Recommended) ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint to confirm the service is running."""
    try:
        if pinecone_index:
            stats = await asyncio.to_thread(pinecone_index.describe_index_stats)
            return {"status": "ok", "message": "Service and Pinecone connected.", "pinecone_stats": stats.to_dict()}
        else:
            return {"status": "warning", "message": "Service running, but Pinecone index not initialized."}
    except Exception as e:
        return {"status": "error", "message": f"Service running, but Pinecone connectivity issue: {e}"}
