# Dockerfile - Single Multi-Stage Build for React + Vite + FastAPI

# =====================================
# Stage 1: Build React Frontend with Vite
# =====================================
FROM node:18 AS frontend-build

WORKDIR /frontend

# Copy package files for better Docker layer caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production --silent

# Copy frontend source code
COPY frontend/ .

# Build arguments for environment configuration
ARG VITE_BACKEND_URL=http://localhost:8001
ARG VITE_API_KEY=sk-default
ARG VITE_AUTO_AUTHENTICATE=false
ARG VITE_DEBUG=false

# Set environment variables for Vite build
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL
ENV VITE_API_KEY=$VITE_API_KEY  
ENV VITE_AUTO_AUTHENTICATE=$VITE_AUTO_AUTHENTICATE
ENV VITE_DEBUG=$VITE_DEBUG

# Build the React application with Vite
RUN npm run build

# Verify build completed successfully
RUN ls -la dist/ && test -f dist/index.html || (echo "❌ Vite build failed" && exit 1)

# =====================================
# Stage 2: Python Dependencies
# =====================================
FROM python:3.11-slim AS python-deps

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --user --no-cache-dir -r /tmp/requirements.txt

# =====================================
# Stage 3: Production Runtime
# =====================================
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=/home/appuser/.local/bin:$PATH

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from python-deps stage
COPY --from=python-deps /root/.local /home/appuser/.local

# Copy backend application files
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser requirements.txt ./

# Copy any additional Python files (if they exist)
COPY --chown=appuser:appuser *.py ./

# Copy built React frontend from frontend-build stage
COPY --from=frontend-build --chown=appuser:appuser /frontend/dist ./static

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/uploads && \
    chown -R appuser:appuser /app

# Copy and make startup script executable
COPY --chown=appuser:appuser start.sh ./
RUN chmod +x start.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start the application
CMD ["./start.sh"]
