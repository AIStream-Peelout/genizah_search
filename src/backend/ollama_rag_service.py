"""
Ollama RAG Service for chat with bibliography search integration.
Uses Cloudflare Access authentication to connect to Ollama endpoint.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import httpx
import json
import dotenv
dotenv.load_dotenv()

from search_bibliography import bibliography_search_service

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat with RAG"""
    message: str = Field(..., min_length=1, max_length=2000, description="User's chat message")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None, 
        description="Previous conversation messages for context"
    )
    num_bibliography_results: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="Number of bibliography results to retrieve for context"
    )
    model: str = Field(
        default="gemma3:27b",
        description="Ollama model to use"
    )


class ChatResponse(BaseModel):
    """Response from chat"""
    message: str = Field(..., description="Assistant's response")
    bibliography_context: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Bibliography results used as context"
    )
    model_used: str = Field(..., description="Model that generated the response")


class OllamaRAGService:
    """Service for RAG-based chat using Ollama with bibliography search"""

    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_URL", "https://ollama.cairogenizah.ai")
        self.cf_authorization = os.getenv("CF_AUTHORIZATION")
        self.cf_client_id = os.getenv("CF-Access-Client-Id")
        self.cf_client_secret = os.getenv("CF-Access-Client-Secret")
        
        if not all([self.cf_authorization, self.cf_client_id, self.cf_client_secret]):
            logger.warning(
                "Missing Cloudflare Access credentials. Chat functionality may not work. "
                "Please set CF_AUTHORIZATION, CF_ACCESS_CLIENT_ID, and CF_ACCESS_CLIENT_SECRET in .env"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get Cloudflare Access headers for authentication"""
        return {
            "CF-Access-Client-Id": self.cf_client_id or "",
            "CF-Access-Client-Secret": self.cf_client_secret or "",
        }

    def _get_cookies(self) -> Dict[str, str]:
        """Get Cloudflare Access cookies for authentication"""
        if self.cf_authorization:
            return {
                "CF_Authorization": self.cf_authorization
            }
        return {}

    async def _search_bibliography(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search bibliography for relevant context"""
        try:
            from search_bibliography import BibliographyHybridSearchRequest
            
            search_request = BibliographyHybridSearchRequest(
                query=query,
                semanticWeight=70,
                keywordWeight=30,
                num_results=num_results,
                page=1
            )
            
            search_response = await bibliography_search_service.search_hybrid(search_request)
            
            # Format results for context
            context_results = []
            for result in search_response.results:
                context_results.append({
                    "doc_id": result.doc_id,
                    "description": result.description,
                    "full_text": result.full_text,
                    "shelf_marks_mentioned": result.shelf_marks_mentioned,
                    "similarity_score": result.similarity_score
                })
            
            return context_results
        except Exception as e:
            logger.error(f"Failed to search bibliography for RAG context: {e}")
            return []

    def _build_rag_prompt(
        self, 
        user_message: str, 
        bibliography_context: List[Dict[str, Any]],
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Build RAG prompt with bibliography context"""
        
        # Build context section from bibliography results
        context_sections = []
        for i, bib in enumerate(bibliography_context, 1):
            context_parts = []
            if bib.get("description"):
                context_parts.append(f"Description: {bib['description']}")
            if bib.get("full_text"):
                # Truncate full_text if too long
                full_text = bib['full_text']
                if len(full_text) > 500:
                    full_text = full_text[:500] + "..."
                context_parts.append(f"Text: {full_text}")
            if bib.get("shelf_marks_mentioned"):
                shelfmarks = ", ".join(bib['shelf_marks_mentioned'][:5])  # Limit shelfmarks
                context_parts.append(f"Shelfmarks: {shelfmarks}")
            
            if context_parts:
                context_sections.append(f"Reference {i}:\n" + "\n".join(context_parts))
        
        context_block = "\n\n".join(context_sections) if context_sections else "No specific references found."
        
        # Build system prompt
        system_prompt = """You are Judaic Studies PhD AI Assistant specialized in the Cairo Genizah collection, a historical archive of more than 400,000 medieval Jewish manuscripts.

You have access to scholarly bibliography references about the Genizah collection. When answering questions, use the provided context from these references to give accurate, well-informed answers. Always cite specific references when possible. When 
possible give direct quotes from the source material

If the context doesn't contain relevant information, you can still provide general knowledge about the Cairo Genizah, but make it clear when you're doing so."""
        
        # Build messages for Ollama
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add bibliography context as a system message
        if context_sections:
            messages.append({
                "role": "system",
                "content": f"Bibliography Context:\n\n{context_block}\n\nUse this context to answer the user's question about the Cairo Genizah collection."
            })
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Perform RAG chat with Ollama"""
        try:
            # Step 1: Search bibliography for relevant context
            logger.info(f"Searching bibliography for query: {request.message[:100]}...")
            bibliography_context = await self._search_bibliography(
                request.message, 
                request.num_bibliography_results
            )
            
            # Step 2: Build RAG prompt
            messages = self._build_rag_prompt(
                request.message,
                bibliography_context,
                request.conversation_history
            )
            
            # Step 3: Call Ollama API
            ollama_url = f"{self.ollama_base_url}/api/chat"
            
            payload = {
                "model": request.model,
                "messages": messages,
                "stream": False
            }
            
            logger.info(f"Calling Ollama API at {ollama_url} with model {request.model}")
            logger.info(f"Calling Ollama with the headers")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    ollama_url,
                    headers=self._get_headers(),
                    cookies=self._get_cookies(),
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            # Extract assistant's message
            assistant_message = result.get("message", {}).get("content", "")
            
            if not assistant_message:
                assistant_message = "I apologize, but I couldn't generate a response. Please try again."
            
            return ChatResponse(
                message=assistant_message,
                bibliography_context=bibliography_context if bibliography_context else None,
                model_used=request.model
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Failed to communicate with Ollama API: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Ollama API request error: {e}")
            raise Exception(f"Network error connecting to Ollama API: {str(e)}")
        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            raise Exception(f"Chat error: {str(e)}")

    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            models_url = f"{self.ollama_base_url}/api/tags"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    models_url,
                    headers=self._get_headers(),
                    cookies=self._get_cookies()
                )
                response.raise_for_status()
                result = response.json()
            
            models = [model.get("name", "") for model in result.get("models", [])]
            return [m for m in models if m]  # Filter out empty names
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            # Return default models if API call fails
            return ["llama3.2", "llama3", "mistral", "qwen2"]


# Global instance
ollama_rag_service = OllamaRAGService()

