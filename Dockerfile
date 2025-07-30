# Use a stable, lightweight Python base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the working directory
COPY . .

# --- START DIAGNOSTIC SECTION ---
# This layer is added specifically to force a cache bust and print directory contents.
# The 'date' command helps ensure this RUN command is unique each time.
RUN echo "--- DIAGNOSTIC: Checking copied files at $(date) ---" && \
    echo "--- Contents of /app ---" && \
    ls -l /app && \
    echo "--- Contents of /app/utils ---" && \
    ls -l /app/utils && \
    echo "--- END DIAGNOSTIC ---"

# Explicitly set the PYTHONPATH to include the current working directory.
ENV PYTHONPATH /app

# Expose the port FastAPI listens on
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
