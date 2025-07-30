# Use a stable, lightweight Python base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first and install Python dependencies
COPY requirements.txt .

# Ensure old pinecone-client is removed, then install all dependencies
RUN pip uninstall -y pinecone-client || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the working directory
COPY . .

# Explicitly set the PYTHONPATH to include the current working directory.
ENV PYTHONPATH /app

# Expose the port FastAPI listens on. Render will detect this.
# This must match the port Uvicorn listens on in the CMD.
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn.
# Uvicorn will listen on port 8000 inside the container.
# Render's system will then route external traffic to this EXPOSEd port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
