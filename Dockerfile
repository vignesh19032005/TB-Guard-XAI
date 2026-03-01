FROM python:3.10-slim

# Install system dependencies needed for OpenCV, PyTorch, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (for faster caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose port (Hugging Face Spaces routes external traffic to 7860)
EXPOSE 7860

# Command to run the Fastapi app
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "7860"]
