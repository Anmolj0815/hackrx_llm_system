# Use a stable, lightweight Python base image (Python 3.11 confirmed to build)
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the working directory
COPY . .

# Expose the port FastAPI listens on
EXPOSE 8000

# === FINAL, DIRECT RUN COMMAND ===
# This CMD directly executes uvicorn. Docker/Python handles PYTHONPATH correctly
# when the app is launched this way from the WORKDIR.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
