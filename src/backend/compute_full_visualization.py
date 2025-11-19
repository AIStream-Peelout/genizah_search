import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from search_service import search_service
from visualization_service import visualization_service

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'full_index_visualization.json')

async def fetch_all_embeddings():
    """Fetch all documents with embeddings from Elasticsearch"""
    logger.info("Fetching all documents with embeddings from Elasticsearch...")
    
    # We'll use the scan helper from elasticsearch to get all documents
    # but search_service doesn't expose it directly. 
    # We'll use a large scroll or pagination if needed, but for now let's try to get a large number
    # assuming the collection isn't massive (e.g. < 50k). 
    # If it is massive, we should use scroll.
    
    # Let's use the underlying ES client from search_service
    es = search_service.es
    index_name = search_service.index_name
    
    query = {
        "bool": {
            "must": [
                {"exists": {"field": "embedding_vector"}}
            ]
        }
    }
    
    # Use scroll to fetch all results
    documents = []
    
    # Initial search
    resp = es.search(
        index=index_name,
        query=query,
        scroll='2m',
        size=1000,
        _source=["doc_id", "embedding_vector", "collection", "document_type", "language", "main_language", "period", "title", "description"]
    )
    
    old_scroll_id = resp['_scroll_id']
    hits = resp['hits']['hits']
    
    while len(hits):
        for hit in hits:
            source = hit['_source']
            if 'embedding_vector' in source and source['embedding_vector']:
                # Extract minimal metadata needed for visualization
                metadata = {
                    'doc_id': source.get('doc_id', hit['_id']),
                    'collection': source.get('collection', 'Unknown'),
                    'document_type': source.get('document_type', 'Unknown'),
                    'language': source.get('language') or source.get('main_language') or 'Unknown',
                    'period': source.get('period', 'Unknown'),
                    'title': source.get('title', f"Document {source.get('doc_id', hit['_id'])}")
                }
                
                documents.append({
                    'doc_id': source.get('doc_id', hit['_id']),
                    'embedding': source['embedding_vector'],
                    'metadata': metadata
                })
        
        logger.info(f"Fetched {len(documents)} documents so far...")
        
        # Scroll to next page
        try:
            resp = es.scroll(
                scroll_id=old_scroll_id,
                scroll='2m'
            )
            old_scroll_id = resp['_scroll_id']
            hits = resp['hits']['hits']
        except Exception as e:
            logger.error(f"Error during scroll: {e}")
            break
            
    logger.info(f"Total documents fetched: {len(documents)}")
    return documents

def compute_visualizations(documents):
    """Compute T-SNE and UMAP coordinates"""
    embeddings = [doc['embedding'] for doc in documents]
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'count': len(documents),
        'documents': []
    }
    
    # Compute T-SNE
    logger.info("Computing T-SNE...")
    tsne_coords, _ = visualization_service.perform_tsne(
        embeddings, 
        perplexity=min(30, len(embeddings) // 10),
        n_iter=1000,
        store_model=False
    )
    
    # Compute UMAP
    logger.info("Computing UMAP...")
    umap_coords, _ = visualization_service.perform_umap(
        embeddings,
        n_neighbors=15,
        min_dist=0.1,
        store_model=False
    )
    
    # Combine results
    for i, doc in enumerate(documents):
        doc_entry = {
            'doc_id': doc['doc_id'],
            'metadata': doc['metadata'],
            'tsne': tsne_coords[i],
            'umap': umap_coords[i]
        }
        results['documents'].append(doc_entry)
        
    return results

def save_results(results):
    """Save results to JSON file"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f)
        
    logger.info(f"Saved visualization data to {OUTPUT_FILE}")

async def main():
    try:
        documents = await fetch_all_embeddings()
        if not documents:
            logger.warning("No documents found with embeddings.")
            return
            
        results = compute_visualizations(documents)
        save_results(results)
        logger.info("Full index visualization computation completed successfully.")
        
    except Exception as e:
        logger.error(f"Failed to compute visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
