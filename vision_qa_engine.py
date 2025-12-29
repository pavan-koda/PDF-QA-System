"""
Vision QA Engine with Llama 3.2-Vision
Handles question-answering using visual understanding of PDFs
Integrates with Ollama for local Llama 3.2-Vision inference
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
import requests
from PIL import Image
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VisionQAEngine:
    """
    QA Engine using Llama 3.2-Vision for multimodal understanding.
    Combines visual page analysis with text extraction.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.2-vision:11b",
        chroma_persist_dir: str = "chroma_db",
        use_colpali: bool = True
    ):
        """
        Initialize Vision QA Engine.

        Args:
            ollama_url: URL for Ollama API server
            model_name: Llama vision model name in Ollama
            chroma_persist_dir: Directory for ChromaDB persistence
            use_colpali: Use ColPali for visual retrieval
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.use_colpali = use_colpali

        logger.info(f"Initializing Vision QA Engine with {model_name}")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize ColPali if enabled
        if use_colpali:
            try:
                from colpali_retriever import ColPaliRetriever
                self.colpali = ColPaliRetriever()
                logger.info("ColPali retriever initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ColPali: {str(e)}")
                self.colpali = None
        else:
            self.colpali = None

        # Check Ollama connection
        self._check_ollama_connection()

    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                logger.info("Ollama server is running")

                # Check if model is available
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                if self.model_name in model_names:
                    logger.info(f"Model {self.model_name} is available")
                    return True
                else:
                    logger.warning(f"Model {self.model_name} not found in Ollama")
                    logger.info(f"Available models: {model_names}")
                    logger.info(f"Run: ollama pull {self.model_name}")
                    return False

            return False

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            logger.info(f"Make sure Ollama is running: ollama serve")
            return False

    def create_collection(
        self,
        session_id: str,
        page_images: List[str],
        page_texts: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Create ChromaDB collection with page data.

        Args:
            session_id: Unique session identifier
            page_images: List of page image paths
            page_texts: List of page text data
            metadata: PDF metadata

        Returns:
            True if successful
        """
        try:
            # Create or get collection
            collection_name = f"pdf_{session_id}"

            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass

            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"session_id": session_id, **metadata}
            )

            # Add pages to collection
            documents = []
            metadatas = []
            ids = []

            for i, (img_path, text_data) in enumerate(zip(page_images, page_texts)):
                page_num = i + 1

                # Combine text and image path
                doc_text = text_data.get('text', '')

                documents.append(doc_text if doc_text else f"Page {page_num}")
                metadatas.append({
                    'page': page_num,
                    'image_path': img_path,
                    'has_text': len(doc_text) > 0,
                    'text_length': len(doc_text)
                })
                ids.append(f"page_{page_num}")

            # Add to ChromaDB
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Created ChromaDB collection with {len(documents)} pages")

            # Create ColPali index if enabled
            if self.colpali:
                self.colpali.create_index(page_images, session_id)

            return True

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False

    def query_with_vision(
        self,
        query: str,
        image_path: str,
        context: str = "",
        max_tokens: int = 2000  # Increased for longer, comprehensive answers
    ) -> str:
        """
        Ask question about an image using Llama 3.2-Vision.

        Args:
            query: Question to ask
            image_path: Path to image
            context: Additional text context
            max_tokens: Maximum response length (up to 2000 for detailed answers)

        Returns:
            Answer from vision model
        """
        try:
            # Resize image for faster processing (max 1024px on longest side)
            img = Image.open(image_path)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Save resized image temporarily
                import io
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                image_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            else:
                # Use original image
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

            # Build detailed prompt for comprehensive answers
            if context:
                prompt = f"""You are analyzing a page from a PDF document. Provide a comprehensive, detailed answer.

Document Context:
{context}

Question: {query}

Instructions:
- Provide a thorough, detailed answer (aim for 3-5 paragraphs if the question requires it)
- If you see diagrams, charts, or images in the page, describe them in detail
- Explain visual elements like flowcharts, architecture diagrams, data visualizations, tables
- Include specific details from both the image and text
- Use clear, structured formatting when appropriate
- Be comprehensive but focused on answering the question

Answer:"""
            else:
                prompt = f"""Analyze this image from a PDF and answer the question in detail.

Question: {query}

Instructions:
- Provide a detailed, comprehensive answer
- Describe any visual elements you see (diagrams, charts, images, tables, graphs)
- Include specific details you observe
- Structure your answer clearly
- Aim for a thorough explanation

Answer:"""

            # Call Ollama API with optimizations for speed
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 2048,  # Smaller context window for faster processing
                    "num_thread": 8,  # Use 8 CPU threads
                    "num_gpu": 0  # CPU only (set to 1 if you have GPU)
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=None  # No timeout - let vision model take as long as needed
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Error querying vision model: {str(e)}")
            return ""

    def answer_question(
        self,
        question: str,
        session_id: str,
        top_k: int = 5,
        use_vision: bool = True,
        use_text_context: bool = True,
        return_images: bool = False
    ):
        """
        Answer question using vision and text understanding.

        Args:
            question: User's question
            session_id: Session identifier
            top_k: Number of top pages to consider
            use_vision: Use vision model for answer
            use_text_context: Include text context
            return_images: Return extracted images with answer

        Returns:
            Answer string or dict with answer and images if return_images=True
        """
        try:
            # Retrieve relevant pages
            relevant_pages = self._retrieve_pages(question, session_id, top_k)

            if not relevant_pages:
                return "I couldn't find relevant information in the document."

            # Get best page
            best_page = relevant_pages[0]
            page_num = best_page['page']
            image_path = best_page['image_path']

            logger.info(f"Using page {page_num} for answer (score: {best_page.get('score', 0):.3f})")

            # Build context from text
            context = ""
            if use_text_context:
                collection_name = f"pdf_{session_id}"
                collection = self.chroma_client.get_collection(collection_name)

                # Get text for top pages
                page_ids = [f"page_{p['page']}" for p in relevant_pages[:3]]
                results = collection.get(ids=page_ids)

                for doc in results['documents']:
                    if doc:
                        context += doc + "\n\n"

            # Smart detection: Decide if we need vision or text is enough
            has_text = len(context.strip()) > 50  # Page has meaningful text content

            # Check if question asks about visual elements
            vision_keywords = ['image', 'picture', 'photo', 'diagram', 'chart', 'graph',
                             'table', 'figure', 'illustration', 'drawing', 'visual', 'show',
                             'look', 'see', 'display']
            asks_about_visuals = any(keyword in question.lower() for keyword in vision_keywords)

            # Use vision ONLY if:
            # 1. Page has NO text (scanned PDF/image) - MUST use vision
            # 2. Question asks about visual elements - use vision to see diagrams/charts
            needs_vision = use_vision and (not has_text or asks_about_visuals)

            # Collect extracted images from the page if requested
            extracted_images = []
            if return_images:
                # Look for embedded images extracted from this page
                from pathlib import Path
                session_dir = Path("data") / session_id
                embedded_dir = session_dir / "embedded_images"

                logger.info(f"Looking for images in: {embedded_dir}")

                if embedded_dir.exists():
                    # Get images from the relevant page (all formats)
                    import glob
                    pattern = str(embedded_dir / f"page_{page_num:04d}_img_*.*")
                    page_images = glob.glob(pattern)

                    logger.info(f"Found {len(page_images)} images for page {page_num}")

                    for img_path in page_images:
                        # Convert to relative path for serving
                        img_name = Path(img_path).name
                        rel_path = f"/data/{session_id}/embedded_images/{img_name}"
                        extracted_images.append(rel_path)
                        logger.info(f"Added image: {rel_path}")
                else:
                    logger.warning(f"Embedded images directory does not exist: {embedded_dir}")

            # Decide mode
            if not has_text:
                logger.info(f"Page has NO TEXT (scanned/image PDF) - Using VISION mode (required)")
                answer = self.query_with_vision(question, image_path, context)
                final_answer = answer if answer else "I couldn't generate an answer."

                if return_images:
                    return {'answer': final_answer, 'images': extracted_images, 'page': int(page_num)}
                return final_answer

            elif asks_about_visuals and use_vision:
                logger.info(f"Question asks about visuals - Using VISION mode")
                answer = self.query_with_vision(question, image_path, context)

                # If vision fails, fall back to text
                if not answer and context:
                    logger.info(f"Vision failed, falling back to text-only")
                    answer = self._text_only_answer(question, context)

                final_answer = answer if answer else "I couldn't generate an answer."

                if return_images:
                    return {'answer': final_answer, 'images': extracted_images, 'page': int(page_num)}
                return final_answer

            else:
                # Text-only mode (FAST!) - page has text and question is about text
                logger.info(f"Page has text, using TEXT-ONLY mode - FAST (no vision needed)")
                final_answer = self._text_only_answer(question, context)

                if return_images:
                    return {'answer': final_answer, 'images': extracted_images, 'page': int(page_num)}
                return final_answer

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error: {str(e)}"

    def _retrieve_pages(
        self,
        query: str,
        session_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant pages using ColPali or ChromaDB.

        Args:
            query: Search query
            session_id: Session identifier
            top_k: Number of results

        Returns:
            List of page results
        """
        try:
            # Try ColPali first for visual retrieval
            if self.colpali:
                results = self.colpali.search(query, session_id, top_k=top_k)
                if results:
                    return results

            # Fallback to ChromaDB text search
            collection_name = f"pdf_{session_id}"
            collection = self.chroma_client.get_collection(collection_name)

            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )

            pages = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                pages.append({
                    'page': metadata['page'],
                    'image_path': metadata['image_path'],
                    'score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    'text': results['documents'][0][i]
                })

            return pages

        except Exception as e:
            logger.error(f"Error retrieving pages: {str(e)}")
            return []

    def _text_only_answer(self, question: str, context: str) -> str:
        """
        Generate answer using text context only (no vision).

        Args:
            question: Question
            context: Text context

        Returns:
            Answer
        """
        try:
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

            payload = {
                "model": self.model_name,  # Use same vision model (can answer text too)
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 500,
                    "temperature": 0.7,
                    "num_ctx": 2048,  # Smaller context for faster response
                    "num_thread": 8
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=None  # No timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()

            return ""

        except Exception as e:
            logger.error(f"Error in text-only answer: {str(e)}")
            return ""

    def cleanup_session(self, session_id: str):
        """Clean up session data."""
        try:
            collection_name = f"pdf_{session_id}"
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Cleaned up session {session_id}")
        except Exception as e:
            logger.warning(f"Error cleaning up session: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = VisionQAEngine()
    print("Vision QA Engine initialized")
