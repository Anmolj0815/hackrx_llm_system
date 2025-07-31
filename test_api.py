import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file for local execution
load_dotenv()

# --- Configuration for API Call ---
# Your deployed Render API's base URL
# Make sure to replace this with your actual, live Render domain!
RENDER_API_BASE_URL = os.getenv("RENDER_API_BASE_URL", "https://hackrx-llm-system.onrender.com")

# Your API authentication token
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# The full endpoint URL
API_ENDPOINT = f"{RENDER_API_BASE_URL}/hackrx/run"

# --- Test Query Data ---
# This is the same structure as the contest platform will send
test_payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# --- API Request Headers ---
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {API_AUTH_TOKEN}"
}

def test_deployed_api():
    """Sends a test query to the deployed FastAPI application."""
    if not API_AUTH_TOKEN:
        print("Error: API_AUTH_TOKEN not set in your local .env file. Please check.")
        return

    print(f"Sending POST request to: {API_ENDPOINT}")
    print(f"Using Authorization: Bearer {API_AUTH_TOKEN[:5]}... (showing first 5 chars)")
    print(f"Payload: documents={test_payload['documents']} with {len(test_payload['questions'])} questions.")

    try:
        response = requests.post(API_ENDPOINT, json=test_payload, headers=headers, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        print("\n--- API Response ---")
        print(response.json()) # Print the JSON response
        print(f"\nStatus Code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"\n--- API Request Failed ---")
        print(f"An error occurred during the API request: {e}")
        if e.response is not None:
            print(f"Response Status Code: {e.response.status_code}")
            try:
                print(f"Response Body: {e.response.json()}")
            except ValueError:
                print(f"Response Body: {e.response.text}")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_deployed_api()
