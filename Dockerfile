# Use a stable, lightweight Python base image (Python 3.11 is confirmed to build)
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the working directory
COPY . .

# Explicitly set the PYTHONPATH to include the current working directory.
ENV PYTHONPATH /app

# Expose the port FastAPI listens on
EXPOSE 8000

# === NEW DIAGNOSTIC AND RUN COMMAND ===
# Use ENTRYPOINT with a shell script to show debug info before starting uvicorn
ENTRYPOINT ["/bin/bash", "-c"]
CMD [ \
    "echo '--- Contents of /app ---';", \
    "ls -l /app;", \
    "echo '--- Contents of /app/utils ---';", \
    "ls -l /app/utils;", \
    "echo '--- Python sys.path ---';", \
    "python -c 'import sys; for p in sys.path: print(p)';", \
    "echo '--- Starting Uvicorn ---';", \
    "exec uvicorn main:app --host 0.0.0.0 --port 8000" \
]
