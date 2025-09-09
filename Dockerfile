FROM nikolaik/python-nodejs:python3.11-nodejs18

WORKDIR /app

# Install dependencies (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Change working directory to where the app code is located
WORKDIR /app/app

# Environment variables (all optional)
# DB_URI: PostgreSQL connection string (falls back to MemorySaver if not provided)
# Example: postgresql://user:password@host:port/database
ENV DB_URI=""

# Expose port
EXPOSE 8000

CMD ["bash", "-c", "if [ ! -z \"$DB_URI\" ]; then python init_pg_checkpointer.py; fi && uvicorn main:app --host 0.0.0.0 --port 8000"]
