# Cairo Genizah AI Website

This is the core code for the [Cairo Genizah AI Project](https://cairogenizah.ai). This is the only web application to support true semantic search of the Cairo Genizah. This project works in conjunction with [Historic Document Analysis](https://github.com/AIStream-Peelout/historical-document-analysis). The code to create the Elasticsearch indices and embeddings is housed there. 

## Setup 
Principally, this is a React and Python based web application that relies on Elasticsearch for search and Neo4j for graph database. 

For local development/deployment you need to have Docker Desktop installed. We are also working on a Kubernetes setup. This setup was tested on an Apple with Silicon.

1. Create a .env file in the `src/backend/` directory. The `.env` should include the following:
   2. ```aiignore
      ELASTICSEARCH_USER=
   ```
      
2. Run `docker compose-up `


## Frontend 
Code for the frontend is in the `frontend` directory. It is a React application that uses the [Mirador](https://github.com/IIIF/mirador) library to display images of the Cairo Genizah. 