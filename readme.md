# Cairo Genizah AI Website

This is the core code for the [Cairo Genizah AI Project](https://cairogenizah.ai). This is the only web application to support true semantic search of the Cairo Genizah. This project works in conjunction with [Historic Document Analysis](https://github.com/AIStream-Peelout/historical-document-analysis). The code to create the Elasticsearch and Neo4j indices and embeddings is housed there. 

## Setup

Principally, this is a React and Python based web application that relies on Elasticsearch for search and Neo4j for graph database.

For local development/deployment you need to have Docker Desktop installed. We are also working on a Kubernetes setup. This setup was tested on an Apple with Silicon.

1. Copy `.env.example` to `.env` at the repository root and replace the
   placeholders needed by your environment. This file supplies the shared
   Docker Compose settings, including Neo4j and Elasticsearch credentials.

2. Optionally create `src/backend/.env` for backend-specific deployment
   overrides. Docker Compose ignores this file when it is absent, so clean
   checkouts and CI can use the defaults and root `.env`. When the backend uses
   an external Elasticsearch instance or an LM Studio server, configure the
   applicable values:

   ```dotenv
   # Elasticsearch connection and index
   ELASTICSEARCH_HOST=elastic.yoursite.example
   ELASTICSEARCH_PORT=443
   ELASTICSEARCH_USER=your_username
   ELASTICSEARCH_PASSWORD=your_password
   ELASTICSEARCH_INDEX=cairo_genizah_text_only_v1.0.1

   # LM Studio and the models used by agentic search
   LLM_STUDIO_URL=http://host.docker.internal:1234
   ROUTER_MODEL=qwen/qwen3-4b-2507
   SYNTHESIS_MODEL=your_synthesis_model
   VERIFICATION_MODEL=your_verification_model

   # Optional runtime tuning
   LM_STUDIO_MAX_CONCURRENCY=1
   LM_STUDIO_MODEL_TTL=3600
   LM_STUDIO_REQUEST_TIMEOUT=300
   SYNTHESIS_MAX_TOKENS=16384
   ```

   Values in `src/backend/.env` override duplicate values from the root `.env`
   for the backend container. Keep both files untracked and never commit real
   credentials. `CHAT_API_KEY` may be added to require authentication for chat
   endpoints; without it, local chat remains unauthenticated. `EVAL_API_KEY`
   enables the private evaluation endpoint and is not needed for normal startup.

3. Run `docker compose up -d`.


## Frontend 
Code for the frontend is in the `frontend` directory. It is a React application that uses the [Mirador](https://github.com/IIIF/mirador) library to display images of the Cairo Genizah. 
