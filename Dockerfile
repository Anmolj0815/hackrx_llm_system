FROM python:3.11-slim-buster
WORKDIR /app
COPY requirements.txt .

# --- ADD THIS LINE TO ENSURE CLEANUP ---
RUN pip uninstall -y pinecone-client || true && \
    pip install --no-cache-dir -r requirements.txt
# --- END ADDITION ---

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
