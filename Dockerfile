FROM python:3.9-slim

WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
COPY app/frontend /app/frontend
COPY ./documents ./documents

# Expose application port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]