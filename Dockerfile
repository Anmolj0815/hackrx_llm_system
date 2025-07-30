# Use a stable, lightweight Python base image
FROM python:3.12-slim-buster

# Set the working directory inside the container
# This is where your code will live. It must match your git repo's root structure.
WORKDIR /app

# Copy requirements.txt and install Python dependencies first
# This layer is cached efficiently if requirements don't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the working directory
# This includes main.py, utils/ directory, ingest_documents.py, etc.
COPY . .

# Explicitly set the PYTHONPATH to include the current working directory.
# This tells Python to look for modules and packages (like 'utils') here.
ENV PYTHONPATH /app

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn
# The 'CMD' instruction defines the default command executed when the container starts.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
