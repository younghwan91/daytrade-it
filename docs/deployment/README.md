# GPT Quant v2 - Deployment Guide

This guide covers local development setup, Docker deployment, and production considerations for the GPT Quant v2 trading system.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Considerations](#production-considerations)
- [API Keys Setup](#api-keys-setup)

---

## Local Development

### Prerequisites

- **Python 3.12+** - Required for the application runtime
- **uv** - Fast Python package installer and resolver ([installation](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker & Docker Compose** - For running TimescaleDB and Redis services
- **Git** - For cloning the repository

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt_quant_v2.git
cd gpt_quant_v2

# Create virtual environment and install dependencies with uv
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Environment Variables

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration. Key variables include:

```bash
# ============================================================
# Database Configuration
# ============================================================
GPTQ_DATABASE_URL=postgresql+asyncpg://gptquant:gptquant@localhost:5432/gptquant
GPTQ_DATABASE_ECHO=false

# ============================================================
# Redis Configuration
# ============================================================
GPTQ_REDIS_URL=redis://localhost:6379/0
GPTQ_REDIS_MAX_CONNECTIONS=10

# ============================================================
# API Keys (Required - see API Keys Setup section)
# ============================================================
GPTQ_POLYGON_API_KEY=your_polygon_api_key_here
GPTQ_OPENAI_API_KEY=your_openai_api_key_here
GPTQ_TIINGO_API_KEY=your_tiingo_api_key_here

# ============================================================
# Server Configuration
# ============================================================
GPTQ_HOST=0.0.0.0
GPTQ_PORT=8000
GPTQ_DEBUG=false
GPTQ_LOG_LEVEL=INFO
```

### Start Services with Docker Compose

Start the required infrastructure services (TimescaleDB and Redis):

```bash
# Navigate to docker directory
cd docker

# Copy and configure docker environment
cp .env.example .env
# Edit .env to set POSTGRES_PASSWORD and optionally REDIS_PASSWORD

# Start services
docker compose up -d

# Verify services are running
docker compose ps

# Check service health
docker compose logs -f
```

The services will be available at:
- **TimescaleDB**: `localhost:5432`
- **Redis**: `localhost:6379`

### Run the Application

```bash
# From project root directory
cd /path/to/gpt_quant_v2

# Run database migrations
alembic upgrade head

# Start the FastAPI server
uvicorn gpt_quant_v2.interfaces.api.main:app --reload --host 0.0.0.0 --port 8000

# Or use the CLI
gptq --help
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

---

## Docker Deployment

### Build Docker Image

Create a `Dockerfile` in the project root (if not present):

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "gpt_quant_v2.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build the image:

```bash
docker build -t gpt-quant-v2:latest .
```

### Docker Compose for Full Stack

Create or extend `docker-compose.yml` for the full application stack:

```yaml
# docker-compose.full.yml
version: "3.8"

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: gpt_quant_app
    restart: unless-stopped
    ports:
      - "${GPTQ_PORT:-8000}:8000"
    environment:
      - GPTQ_DATABASE_URL=postgresql+asyncpg://${POSTGRES_USER:-gpt_quant}:${POSTGRES_PASSWORD}@timescaledb:5432/${POSTGRES_DB:-gpt_quant}
      - GPTQ_REDIS_URL=redis://:${REDIS_PASSWORD:-}@redis:6379/0
      - GPTQ_POLYGON_API_KEY=${GPTQ_POLYGON_API_KEY}
      - GPTQ_OPENAI_API_KEY=${GPTQ_OPENAI_API_KEY}
      - GPTQ_TIINGO_API_KEY=${GPTQ_TIINGO_API_KEY}
      - GPTQ_LOG_LEVEL=${GPTQ_LOG_LEVEL:-INFO}
      - GPTQ_DEBUG=${GPTQ_DEBUG:-false}
    depends_on:
      timescaledb:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - gpt_quant_network
    volumes:
      - app_logs:/app/logs
      - app_models:/app/models

  timescaledb:
    image: timescale/timescaledb:latest-pg15-alpine
    container_name: gpt_quant_timescaledb
    restart: unless-stopped
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-gpt_quant}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}
      POSTGRES_DB: ${POSTGRES_DB:-gpt_quant}
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
      - ./docker/timescaledb/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-gpt_quant} -d ${POSTGRES_DB:-gpt_quant}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    networks:
      - gpt_quant_network

  redis:
    image: redis:7-alpine
    container_name: gpt_quant_redis
    restart: unless-stopped
    ports:
      - "${REDIS_PORT:-6379}:6379"
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --requirepass ${REDIS_PASSWORD:-}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    networks:
      - gpt_quant_network

volumes:
  timescaledb_data:
    name: gpt_quant_timescaledb_data
  redis_data:
    name: gpt_quant_redis_data
  app_logs:
    name: gpt_quant_app_logs
  app_models:
    name: gpt_quant_app_models

networks:
  gpt_quant_network:
    name: gpt_quant_network
    driver: bridge
```

### Environment Configuration

Create a `.env` file for Docker Compose:

```bash
# .env for docker-compose
POSTGRES_USER=gpt_quant
POSTGRES_PASSWORD=secure_password_here
POSTGRES_DB=gpt_quant
POSTGRES_PORT=5432

REDIS_PASSWORD=secure_redis_password
REDIS_PORT=6379

GPTQ_PORT=8000
GPTQ_LOG_LEVEL=INFO
GPTQ_DEBUG=false

# API Keys
GPTQ_POLYGON_API_KEY=your_polygon_key
GPTQ_OPENAI_API_KEY=your_openai_key
GPTQ_TIINGO_API_KEY=your_tiingo_key
```

Deploy the full stack:

```bash
docker compose -f docker-compose.full.yml up -d
```

### Health Checks

All services include health checks:

| Service | Health Check | Interval |
|---------|--------------|----------|
| App | `GET /health` | 30s |
| TimescaleDB | `pg_isready` | 10s |
| Redis | `redis-cli ping` | 10s |

Monitor health status:

```bash
docker compose ps
docker compose logs --tail=100 -f
```

---

## Production Considerations

### Database Migrations (Alembic)

GPT Quant v2 uses Alembic for database schema migrations with async PostgreSQL support.

```bash
# Create a new migration
alembic revision --autogenerate -m "description of changes"

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# View current revision
alembic current
```

**Production migration workflow:**

```bash
# 1. Backup database before migrations
pg_dump -h localhost -U gpt_quant gpt_quant > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Run migrations
alembic upgrade head

# 3. Verify migration success
alembic current
```

For Docker deployments, run migrations as a separate step or init container:

```bash
docker compose exec app alembic upgrade head
```

### Secrets Management

**Never commit secrets to version control.** Use one of these approaches:

1. **Environment Variables** (Development)
   ```bash
   export GPTQ_OPENAI_API_KEY=sk-xxx
   ```

2. **Docker Secrets** (Docker Swarm)
   ```yaml
   secrets:
     openai_api_key:
       external: true
   services:
     app:
       secrets:
         - openai_api_key
   ```

3. **HashiCorp Vault** (Enterprise)
   ```bash
   vault kv get -field=api_key secret/gpt_quant/openai
   ```

4. **AWS Secrets Manager / Azure Key Vault / GCP Secret Manager**
   - Use cloud-native secret management for cloud deployments

**Recommended production setup:**
- Use a secrets manager (Vault, AWS Secrets Manager, etc.)
- Rotate API keys periodically
- Use separate API keys for development/staging/production
- Set up alerts for key usage anomalies

### Logging Configuration

GPT Quant v2 uses `structlog` for structured JSON logging.

```bash
# Environment variables for logging
GPTQ_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
GPTQ_LOG_FORMAT=json         # json or console
GPTQ_LOG_FILE=logs/gptquant.log
```

**Log aggregation recommendations:**

1. **ELK Stack** (Elasticsearch, Logstash, Kibana)
   ```yaml
   # docker-compose addition
   logging:
     driver: "fluentd"
     options:
       fluentd-address: localhost:24224
   ```

2. **Cloud Logging Services**
   - AWS CloudWatch Logs
   - Google Cloud Logging
   - Azure Monitor Logs

3. **Grafana Loki** (lightweight alternative)
   ```yaml
   logging:
     driver: loki
     options:
       loki-url: "http://loki:3100/loki/api/v1/push"
   ```

### Monitoring Recommendations

**Metrics to monitor:**

| Category | Metrics |
|----------|---------|
| Application | Request latency, error rates, active connections |
| Trading | Signals generated, orders executed, P&L |
| Infrastructure | CPU, memory, disk I/O, network |
| Database | Connection pool, query latency, replication lag |
| Redis | Memory usage, cache hit rate, command latency |

**Recommended monitoring stack:**

1. **Prometheus + Grafana**
   ```python
   # Add to FastAPI app
   from prometheus_fastapi_instrumentator import Instrumentator
   Instrumentator().instrument(app).expose(app)
   ```

2. **Application Performance Monitoring (APM)**
   - Datadog
   - New Relic
   - Sentry (for error tracking)

3. **Alerting**
   - Set up alerts for:
     - API error rate > 1%
     - P95 latency > 500ms
     - Database connection failures
     - Trading anomalies (unusual order volumes)

### Scaling Considerations

**Horizontal scaling:**

```yaml
# docker-compose scale
services:
  app:
    deploy:
      replicas: 3
```

**Load balancing with nginx:**

```nginx
upstream gpt_quant {
    least_conn;
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://gpt_quant;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Kubernetes deployment considerations:**
- Use Horizontal Pod Autoscaler (HPA) based on CPU/memory
- Implement pod disruption budgets for high availability
- Use StatefulSets for TimescaleDB if not using managed database
- Consider managed services (AWS RDS, Redis ElastiCache)

**Database scaling:**
- **Read replicas** for query-heavy workloads
- **TimescaleDB compression** for historical data
- **Connection pooling** with PgBouncer
- **Partitioning** for large tables (TimescaleDB hypertables)

---

## API Keys Setup

### Polygon.io

Polygon.io provides real-time and historical market data.

1. **Create an account** at [polygon.io](https://polygon.io/)
2. **Choose a plan:**
   - Free: 5 API calls/minute (development only)
   - Starter: $29/month - 500 calls/minute
   - Developer: $79/month - unlimited calls (recommended for production)
3. **Get your API key** from the Dashboard → API Keys section
4. **Configure:**
   ```bash
   GPTQ_POLYGON_API_KEY=your_polygon_api_key_here
   ```

**Usage in GPT Quant:**
- Real-time stock quotes
- Historical price data (OHLCV)
- Options data
- Market status

### Tiingo

Tiingo provides news and fundamental data.

1. **Create an account** at [tiingo.com](https://www.tiingo.com/)
2. **Get your API token** from Account → API → Token
3. **Free tier includes:**
   - 1,000 requests/day
   - News API access
   - End-of-day data
4. **Configure:**
   ```bash
   GPTQ_TIINGO_API_KEY=your_tiingo_api_key_here
   ```

**Usage in GPT Quant:**
- Financial news data
- News sentiment analysis inputs
- Company fundamentals

### OpenAI

OpenAI API powers the sentiment analysis and AI features.

1. **Create an account** at [platform.openai.com](https://platform.openai.com/)
2. **Add billing** - Pay-as-you-go pricing
3. **Create an API key** from API Keys section
4. **Configure:**
   ```bash
   GPTQ_OPENAI_API_KEY=your_openai_api_key_here
   GPTQ_OPENAI_MODEL=gpt-4o-mini    # Cost-effective for sentiment
   GPTQ_OPENAI_TEMPERATURE=0.1      # Low temp for consistent results
   ```

**Pricing considerations:**
- `gpt-4o-mini`: ~$0.15/1M input tokens, $0.60/1M output tokens (recommended)
- `gpt-4o`: ~$2.50/1M input tokens, $10/1M output tokens
- Set up usage limits in OpenAI dashboard to prevent unexpected costs

**Usage in GPT Quant:**
- News sentiment analysis
- Signal interpretation
- Market context analysis

### Optional: Alpaca (Paper Trading / Live Trading)

For actual order execution:

1. **Create account** at [alpaca.markets](https://alpaca.markets/)
2. **Get API keys** from the dashboard (Paper trading keys for testing)
3. **Configure:**
   ```bash
   GPTQ_BROKER_TYPE=alpaca          # or 'paper' for simulation
   GPTQ_ALPACA_API_KEY=your_key
   GPTQ_ALPACA_SECRET_KEY=your_secret
   GPTQ_ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper for testing
   ```

---

## Quick Reference

```bash
# Development
uv sync && source .venv/bin/activate
docker compose -f docker/docker-compose.yml up -d
alembic upgrade head
uvicorn gpt_quant_v2.interfaces.api.main:app --reload

# Production
docker compose -f docker-compose.full.yml up -d
docker compose exec app alembic upgrade head
docker compose logs -f

# Maintenance
docker compose exec timescaledb pg_dump -U gpt_quant gpt_quant > backup.sql
docker compose restart app
```

---

For more information, see:
- [API Documentation](../api/README.md)
- [Architecture Decision Records](../adr/)
- [System Diagrams](../diagrams/)
