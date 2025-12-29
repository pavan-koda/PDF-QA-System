import logging
import pickle
import os
import re
from typing import List, Optional, Tuple
import numpy as np
import torch
from pathlib import Path

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

try:
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, pipeline,
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    )
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

logger = logging.getLogger(__name__)


class QAEngine:
    """Question-Answering engine using FAISS for retrieval and GPT-2 for generation."""

    def __init__(
        self,
        embedder_model: str = "all-MiniLM-L6-v2",
        gpt2_model: str = "./gpt2-medium",
        data_dir: str = "data",
        use_advanced_qa: bool = False,
        advanced_qa_model: str = "distilbert-base-cased-distilled-squad"
    ):
        """
        Initialize the QA Engine.

        Args:
            embedder_model: Name of the sentence transformer model
            gpt2_model: Path to GPT-2 model (local or HuggingFace)
            data_dir: Directory to store session data
            use_advanced_qa: Use advanced QA model (DistilBERT/RoBERTa) for better answers
            advanced_qa_model: Which advanced QA model to use
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.use_advanced_qa = use_advanced_qa
        self.qa_pipeline = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize models
        self._load_models(embedder_model, gpt2_model, advanced_qa_model)

    def _load_models(self, embedder_model: str, gpt2_model: str, advanced_qa_model: str):
        """Load the embedding and generation models."""
        try:
            logger.info("Loading sentence transformer model...")
            self.embedder = SentenceTransformer(embedder_model)
            logger.info("Sentence transformer loaded successfully")

            # Load advanced QA model if requested
            if self.use_advanced_qa:
                logger.info(f"Loading advanced QA model: {advanced_qa_model}...")
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=advanced_qa_model,
                    tokenizer=advanced_qa_model,
                    device=0 if self.device.type == "cuda" else -1
                )
                logger.info("Advanced QA model loaded successfully")

            # Load generator model only if not "none"
            if gpt2_model and gpt2_model.lower() != "none":
                logger.info(f"Loading generator model: {gpt2_model}...")

                # Determine model path (local or HuggingFace)
                model_path = gpt2_model
                local_path = Path(gpt2_model.replace('/', '--'))

                if local_path.exists():
                    model_path = str(local_path)
                    logger.info(f"Loading from local path: {model_path}")
                elif not Path(gpt2_model).exists():
                    logger.info(f"Local model not found, will download from HuggingFace")

                # Detect model type and load appropriately
                self.is_seq2seq = 't5' in gpt2_model.lower() or 'flan' in gpt2_model.lower()

                # Check if it's GPT-OSS model (special handling)
                self.is_gpt_oss = 'gpt-oss' in gpt2_model.lower()

                # Check if model needs trust_remote_code (Gemma 3, Phi-3, GPT-OSS, etc.)
                needs_trust = 'gemma-3' in gpt2_model.lower() or 'gemma3' in gpt2_model.lower() or self.is_gpt_oss

                if self.is_seq2seq:
                    # T5/FLAN-T5 models (seq2seq)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=needs_trust
                    )
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        trust_remote_code=needs_trust
                    )
                    logger.info("Loaded as Seq2Seq model (T5/FLAN-T5)")
                else:
                    # GPT-2, OPT, Llama, Phi, StableLM, Gemma (causal LM)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=needs_trust
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=needs_trust
                    )
                    logger.info("Loaded as Causal LM model")

                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model.to(self.device)
                self.model.eval()
                logger.info("Generator model loaded successfully")
            else:
                logger.info("Skipping generator model (not needed for extractive mode)")
                self.tokenizer = None
                self.model = None
                self.is_seq2seq = False
                self.is_gpt_oss = False

            self.models_loaded = True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
            raise

    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return self.models_loaded

    def create_index(self, chunks: List[str], session_id: str) -> bool:
        """
        Create FAISS index from text chunks.

        Args:
            chunks: List of text chunks
            session_id: Unique session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks:
                logger.error("Cannot create index from empty chunks")
                return False

            logger.info(f"Creating embeddings for {len(chunks)} chunks...")

            # Create embeddings
            embeddings = self.embedder.encode(
                chunks,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Normalize embeddings for cosine similarity
            embeddings = self._normalize_embeddings(embeddings)

            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            index.add(embeddings)

            # Save chunks and index
            chunks_path = self.data_dir / f"{session_id}_chunks.pkl"
            index_path = self.data_dir / f"{session_id}_index.faiss"

            with open(chunks_path, 'wb') as f:
                pickle.dump(chunks, f)

            faiss.write_index(index, str(index_path))

            logger.info(f"Created index with {len(chunks)} chunks for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def get_relevant_chunks(
        self,
        query: str,
        session_id: str,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Retrieve relevant chunks for a query with similarity scores.

        Args:
            query: User question
            session_id: Session identifier
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score to include chunk

        Returns:
            List of tuples (chunk, score) or None if error
        """
        try:
            # Load session data
            chunks_path = self.data_dir / f"{session_id}_chunks.pkl"
            index_path = self.data_dir / f"{session_id}_index.faiss"

            if not chunks_path.exists() or not index_path.exists():
                logger.error(f"Session data not found for {session_id}")
                return None

            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)

            index = faiss.read_index(str(index_path))

            # Create query embedding
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            query_embedding = self._normalize_embeddings(query_embedding)

            # Search - get more chunks initially
            search_k = min(len(chunks), max(top_k * 2, 10))
            scores, indices = index.search(query_embedding, search_k)

            # Filter by score threshold and get chunks with scores
            relevant_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= score_threshold:
                    relevant_chunks.append((chunks[idx], float(score)))

            # If no chunks meet threshold, take top k anyway
            if not relevant_chunks:
                relevant_chunks = [(chunks[i], float(s)) for s, i in zip(scores[0][:top_k], indices[0][:top_k])]

            logger.info(f"Retrieved {len(relevant_chunks)} chunks for query (threshold: {score_threshold})")
            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return None

    def get_all_chunks(self, session_id: str) -> Optional[str]:
        """
        Get all chunks as a single text (full document context).

        Args:
            session_id: Session identifier

        Returns:
            Full document text or None if error
        """
        try:
            chunks_path = self.data_dir / f"{session_id}_chunks.pkl"

            if not chunks_path.exists():
                logger.error(f"Session data not found for {session_id}")
                return None

            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)

            # Combine all chunks into full text
            full_text = " ".join(chunks)
            logger.info(f"Retrieved full document with {len(chunks)} chunks")
            return full_text

        except Exception as e:
            logger.error(f"Error retrieving full document: {str(e)}")
            return None

    def answer_question(
        self,
        question: str,
        session_id: str,
        use_extractive: bool = True,
        use_full_context: bool = True,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.92
    ) -> Optional[str]:
        """
        Answer a question using retrieval + generation or extraction.

        Args:
            question: User question
            session_id: Session identifier
            use_extractive: If True, return relevant chunks directly (more accurate)
            use_full_context: If True, use full document for QA (better accuracy)
            max_new_tokens: Maximum tokens to generate (if use_extractive=False)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated answer or None if error
        """
        try:
            # Check if this is a conversational/greeting question
            if self._is_conversational_question(question):
                return self._handle_conversational_question(question)

            # Get context - either full document or top chunks
            if use_full_context:
                full_text = self.get_all_chunks(session_id)
                if not full_text:
                    logger.error("Failed to retrieve full document")
                    return None
                context_text = full_text
                relevant_chunks_list = [full_text]  # Treat as single chunk for QA
            else:
                # Get relevant chunks with scores
                relevant_chunks_with_scores = self.get_relevant_chunks(question, session_id, top_k=10)

                if not relevant_chunks_with_scores:
                    logger.error("Failed to retrieve relevant chunks")
                    return None

                # Extract just the chunks (remove scores)
                relevant_chunks_list = [chunk for chunk, score in relevant_chunks_with_scores]
                context_text = " ".join(relevant_chunks_list[:5])

            # Use extractive approach (return actual text from PDF)
            if use_extractive:
                # Try advanced QA model first if available
                if self.use_advanced_qa and self.qa_pipeline:
                    answer = self._answer_with_advanced_qa(context_text, question, use_full_context)
                    if answer:
                        return answer

                # Fall back to extractive approach
                answer = self._format_extractive_answer(relevant_chunks_list, question, use_full_context)
                return answer

            # Use generative approach (GPT-2 generation)
            else:
                # Create prompt
                prompt = self._create_prompt(context_text, question)

                # Generate answer
                answer = self._generate_answer(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return None

    def _is_conversational_question(self, question: str) -> bool:
        """
        Check if the question is conversational/greeting rather than document-related.

        Args:
            question: User's question

        Returns:
            True if conversational, False otherwise
        """
        question_lower = question.lower().strip()

        # Greetings and conversational phrases
        conversational_patterns = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon',
            'good evening', 'how are you', 'whats up', "what's up", 'sup',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you',
            'ok', 'okay', 'yes', 'no', 'cool', 'nice', 'great'
        ]

        # Check if question is just a conversational phrase
        if question_lower in conversational_patterns:
            return True

        # Check if question is very short (less than 3 words) and doesn't have question words
        words = question_lower.split()
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'tell', 'show', 'find', 'get', 'list']

        if len(words) <= 2 and not any(qw in words for qw in question_words):
            return True

        return False

    def _handle_conversational_question(self, question: str) -> str:
        """
        Handle conversational/greeting questions.

        Args:
            question: User's question

        Returns:
            Appropriate conversational response
        """
        question_lower = question.lower().strip()

        if any(greeting in question_lower for greeting in ['hi', 'hello', 'hey', 'greetings']):
            return "Hello! I'm here to help you with questions about your PDF document. Please ask me anything about the document content."

        if 'how are you' in question_lower:
            return "I'm doing well, thank you! I'm ready to answer questions about your PDF document. What would you like to know?"

        if any(thanks in question_lower for thanks in ['thanks', 'thank you']):
            return "You're welcome! Feel free to ask more questions about your document."

        if any(bye in question_lower for bye in ['bye', 'goodbye', 'see you']):
            return "Goodbye! Feel free to upload another document anytime."

        # Default response for other conversational inputs
        return "I'm here to help you understand your PDF document. Please ask me a specific question about the document content."

    def _answer_with_advanced_qa(self, context: str, question: str, use_full_context: bool = False) -> Optional[str]:
        """
        Use advanced QA model (DistilBERT/RoBERTa) to answer the question.

        Args:
            context: Text context (can be full document or chunks)
            question: The user's question
            use_full_context: Whether using full document context

        Returns:
            Answer from the QA model or None
        """
        try:
            # Truncate if too long (BERT models have max length limits)
            max_context_length = 8000 if use_full_context else 4000
            if len(context) > max_context_length:
                # Smart truncation: try to keep relevant parts
                context = self._smart_truncate(context, question, max_context_length)

            # Use the QA pipeline
            result = self.qa_pipeline(question=question, context=context)

            # Check confidence score
            if result['score'] > 0.05:  # Lower threshold for full context
                answer = result['answer']
                score = result['score']

                # Get surrounding context for the answer
                answer_context = self._get_context_around_match(context, answer, 200)

                # Add confidence indicator
                confidence = "High" if score > 0.5 else "Medium" if score > 0.3 else "Low"

                return f"{answer}\n\nContext: {answer_context}\n\n[Confidence: {confidence} ({score:.2f})]"

            return None

        except Exception as e:
            logger.error(f"Error with advanced QA model: {str(e)}")
            return None

    def _preserve_list_formatting(self, text: str) -> str:
        """
        Preserve and enhance list formatting in text.

        Args:
            text: Text that may contain lists

        Returns:
            Text with improved list formatting
        """
        # Detect and format bullet points
        # Common patterns: • - * · ○ □
        text = re.sub(r'\s*[•\-\*·○□]\s+', '\n• ', text)

        # Detect numbered lists (1. 2. 3. or 1) 2) 3))
        text = re.sub(r'\s+(\d+)[.)\]]\s+', r'\n\1. ', text)

        # Detect lettered lists (a. b. c. or a) b) c))
        text = re.sub(r'\s+([a-z])[.)\]]\s+', r'\n\1. ', text)

        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Ensure lists start on new line
        text = re.sub(r'([.!?:])\s*\n•', r'\1\n\n•', text)
        text = re.sub(r'([.!?:])\s*\n(\d+\.)', r'\1\n\n\2', text)

        return text.strip()

    def _smart_truncate(self, text: str, question: str, max_length: int) -> str:
        """
        Smart truncation that tries to keep text relevant to the question.

        Args:
            text: Text to truncate
            question: User's question
            max_length: Maximum character length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Extract keywords from question
        question_words = set(re.findall(r'\w+', question.lower()))
        question_words.discard('what')
        question_words.discard('how')
        question_words.discard('when')
        question_words.discard('where')
        question_words.discard('who')
        question_words.discard('why')
        question_words.discard('the')
        question_words.discard('is')
        question_words.discard('are')

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Score sentences by keyword presence
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for word in question_words if word in sent_lower)
            scored_sentences.append((score, sent))

        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Take top sentences until we hit max_length
        result = []
        current_length = 0
        for score, sent in scored_sentences:
            if current_length + len(sent) <= max_length:
                result.append(sent)
                current_length += len(sent)
            else:
                break

        # If we got nothing, just take first max_length characters
        if not result:
            return text[:max_length]

        return " ".join(result)

    def _format_extractive_answer(self, chunks: List[str], question: str, use_full_context: bool = False) -> str:
        """
        Format relevant chunks as an extractive answer.

        Args:
            chunks: List of relevant text chunks or full document
            question: The user's question
            use_full_context: Whether using full document context

        Returns:
            Formatted answer from the chunks
        """
        question_lower = question.lower()

        # If using full context, work with the entire document
        if use_full_context and chunks:
            full_text = chunks[0] if len(chunks) == 1 else " ".join(chunks)

            # Check if question is asking for specific information
            specific_answer = self._extract_specific_info_from_text(full_text, question_lower)
            if specific_answer:
                return self._preserve_list_formatting(specific_answer)

            # Use smart extraction based on question keywords
            answer = self._extract_relevant_section(full_text, question_lower)
            return self._preserve_list_formatting(answer)

        # Original chunked approach for backward compatibility
        # Check if question is asking for specific information
        specific_answer = self._extract_specific_info(chunks, question_lower)
        if specific_answer:
            return specific_answer

        # Combine top chunks with relevance indicators
        answer_parts = []

        for i, chunk in enumerate(chunks[:5]):  # Use top 5 most relevant chunks
            # Clean up the chunk
            chunk_clean = ' '.join(chunk.split())

            # Add chunk with some context
            if i == 0:
                answer_parts.append(chunk_clean)
            else:
                # Add additional chunks if they seem to add new information
                if not any(chunk_clean in part for part in answer_parts):
                    answer_parts.append(chunk_clean)

        # Join the parts
        answer = "\n\n".join(answer_parts)

        # If answer is too long, truncate intelligently
        if len(answer) > 1200:
            answer = self._smart_truncate(answer, question, 1200)

        return answer.strip()

    def _extract_relevant_section(self, text: str, question_lower: str) -> str:
        """
        Extract the most relevant section from full text based on question.

        Args:
            text: Full document text
            question_lower: Lowercase question

        Returns:
            Most relevant section (concise)
        """
        # Extract keywords from question
        keywords = set(re.findall(r'\w+', question_lower))
        keywords.discard('what')
        keywords.discard('how')
        keywords.discard('when')
        keywords.discard('where')
        keywords.discard('who')
        keywords.discard('why')
        keywords.discard('is')
        keywords.discard('are')
        keywords.discard('the')
        keywords.discard('a')
        keywords.discard('an')
        keywords.discard('about')
        keywords.discard('this')
        keywords.discard('that')

        # If no meaningful keywords, return generic response
        if not keywords or len(keywords) < 2:
            return "Please ask a specific question about the document content (e.g., 'What is the total amount?', 'When is the date?', 'Who is the vendor?')."

        # Split into sentences for more precise extraction
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Score each sentence
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Count keyword occurrences and keyword coverage
            keyword_count = sum(sentence_lower.count(keyword) for keyword in keywords)
            keyword_coverage = sum(1 for keyword in keywords if keyword in sentence_lower)

            # Combined score
            score = keyword_count * 2 + keyword_coverage

            if score > 0:
                scored_sentences.append((score, sentence))

        # Sort by score
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Take top sentences (limit to most relevant)
        if scored_sentences:
            # Get top 3-5 sentences
            top_sentences = [sentence for score, sentence in scored_sentences[:5]]

            # Combine into coherent answer
            answer = " ".join(top_sentences)

            # Truncate if too long
            if len(answer) > 800:
                # Take only top 3 sentences
                answer = " ".join(top_sentences[:3])

            if len(answer) > 800:
                answer = answer[:800] + "..."

            return answer.strip()

        # Fallback: return beginning of document
        return text[:500].strip() + "..."

    def _extract_specific_info_from_text(self, text: str, question_lower: str) -> Optional[str]:
        """
        Extract specific information from full text.

        Args:
            text: Full document text
            question_lower: Lowercase question

        Returns:
            Specific answer if found, None otherwise
        """
        # Amount/price/total questions
        if any(word in question_lower for word in ['amount', 'total', 'price', 'cost', 'payment', 'bill', 'pay']):
            amounts = self._extract_amounts(text)
            if amounts:
                # Find the largest amount (likely the total)
                max_amount = max(amounts, key=lambda x: x[0])

                # Get context around the amount
                context = self._get_context_around_match(text, max_amount[1], 200)

                # Look for all significant amounts
                significant_amounts = [amt for amt in amounts if amt[0] > 100][:5]
                amounts_str = ", ".join([amt[1] for amt in significant_amounts])

                return f"Total/Main Amount: {max_amount[1]}\n\nAll amounts found: {amounts_str}\n\nContext: {context}"

        # Date questions
        if any(word in question_lower for word in ['date', 'when']):
            dates = self._extract_dates(text)
            if dates:
                # Get context around first date
                context = self._get_context_around_match(text, dates[0], 250)
                return f"Dates found: {', '.join(dates)}\n\nContext: {context}"

        # Name/company questions
        if any(word in question_lower for word in ['name', 'company', 'who', 'vendor', 'seller', 'buyer', 'customer']):
            # Look for capitalized words (likely names/companies)
            names = self._extract_names(text)
            if names:
                # Get context from beginning of document (often has company names)
                context = text[:500]
                return f"Names/Companies found: {', '.join(names[:10])}\n\nContext: {context}"

        return None

    def _extract_specific_info(self, chunks: List[str], question_lower: str) -> Optional[str]:
        """
        Extract specific information like amounts, dates, names based on question type.

        Args:
            chunks: List of text chunks
            question_lower: Lowercase question

        Returns:
            Specific answer if found, None otherwise
        """
        combined_text = " ".join(chunks)

        # Amount/price/total questions
        if any(word in question_lower for word in ['amount', 'total', 'price', 'cost', 'payment', 'bill']):
            amounts = self._extract_amounts(combined_text)
            if amounts:
                # Find the largest amount (likely the total)
                max_amount = max(amounts, key=lambda x: x[0])

                # Get context around the amount
                context = self._get_context_around_match(combined_text, max_amount[1], 100)

                return f"Total Amount: {max_amount[1]}\n\nContext: {context}"

        # Date questions
        if any(word in question_lower for word in ['date', 'when']):
            dates = self._extract_dates(combined_text)
            if dates:
                return f"Dates found: {', '.join(dates[:3])}\n\nContext: {chunks[0][:300]}..."

        # Name/company questions
        if any(word in question_lower for word in ['name', 'company', 'who', 'vendor', 'seller']):
            # Look for capitalized words (likely names/companies)
            names = self._extract_names(combined_text)
            if names:
                return f"Names/Companies found: {', '.join(names[:5])}\n\nContext: {chunks[0][:300]}..."

        return None

    def _extract_amounts(self, text: str) -> List[Tuple[float, str]]:
        """Extract monetary amounts from text."""
        # Patterns for amounts: ₹123.45, $123.45, 123.45, Rs. 123, etc.
        patterns = [
            r'(?:₹|Rs\.?|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)',  # ₹1,234.56
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:₹|Rs\.?|INR)',  # 1,234.56 ₹
            r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)',                # $1,234.56
            r'(?:Total|Amount|Grand Total|Net)[:\s]+(?:₹|Rs\.?)?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
        ]

        amounts = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1) if match.lastindex else match.group(0)
                # Remove commas and convert to float
                try:
                    amount_clean = amount_str.replace(',', '')
                    amount_val = float(amount_clean)
                    amounts.append((amount_val, match.group(0)))
                except ValueError:
                    continue

        return sorted(amounts, key=lambda x: x[0], reverse=True)

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        # Common date patterns
        patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',           # 12/31/2023, 12-31-23
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',             # 2023-12-31
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # 31 Dec 2023
        ]

        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)

        return list(set(dates))[:5]  # Return unique dates

    def _extract_names(self, text: str) -> List[str]:
        """Extract potential names/companies (capitalized words)."""
        # Find sequences of capitalized words (likely names/companies)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Pvt\.?|Ltd\.?|Inc\.?|Corp\.?|Limited))?\b'
        matches = re.findall(pattern, text)

        # Filter out common words
        common_words = {'The', 'This', 'That', 'With', 'From', 'To', 'For', 'And', 'Or', 'But'}
        names = [m for m in matches if m not in common_words]

        return list(set(names))[:10]  # Return unique names

    def _get_context_around_match(self, text: str, match: str, context_chars: int = 150) -> str:
        """Get text context around a match."""
        pos = text.find(match)
        if pos == -1:
            return text[:context_chars]

        start = max(0, pos - context_chars)
        end = min(len(text), pos + len(match) + context_chars)

        context = text[start:end].strip()
        return context

    def _create_prompt(self, context: str, question: str) -> str:
        """Create a prompt for the language model."""
        # GPT-OSS models use chat format (harmony format handled by tokenizer)
        if self.is_gpt_oss:
            # For GPT-OSS, return messages list format
            return [
                {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]

        # Standard prompt for other models
        prompt = (
            f"Use the following context to answer the question accurately and concisely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt

    def _generate_answer(
        self,
        prompt,  # Can be str or list (for GPT-OSS)
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.92
    ) -> Optional[str]:
        """
        Generate answer using the loaded generator model.

        Args:
            prompt: Input prompt (str for regular models, list for GPT-OSS)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            Generated text or None if error
        """
        try:
            # Handle GPT-OSS chat format
            if self.is_gpt_oss and isinstance(prompt, list):
                # Apply chat template for GPT-OSS
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
            else:
                # Standard tokenization for other models
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Generate based on model type
            with torch.no_grad():
                if self.is_seq2seq:
                    # T5/FLAN-T5 models generate directly without prompt in output
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        early_stopping=True
                    )
                    # For seq2seq, decode the entire output
                    answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                else:
                    # Causal LM models (GPT-2, OPT, Llama, etc.)
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                    # For causal LM, decode only the generated tokens
                    generated_tokens = output_ids[0][input_ids.shape[-1]:]
                    answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Clean up the answer
            answer = self._clean_answer(answer)

            return answer

        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return None

    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer."""
        # Remove extra whitespace
        answer = ' '.join(answer.split())

        # Stop at first newline (often end of answer)
        if '\n' in answer:
            answer = answer.split('\n')[0]

        # If answer is too short, return as is
        if len(answer) < 10:
            return answer.strip()

        # Truncate at last complete sentence if answer is very long
        if len(answer) > 500:
            sentences = answer.split('. ')
            answer = '. '.join(sentences[:-1]) + '.'

        return answer.strip()

    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up session data.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            chunks_path = self.data_dir / f"{session_id}_chunks.pkl"
            index_path = self.data_dir / f"{session_id}_index.faiss"

            if chunks_path.exists():
                chunks_path.unlink()

            if index_path.exists():
                index_path.unlink()

            logger.info(f"Cleaned up session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up session: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the QA engine
    logging.basicConfig(level=logging.INFO)

    engine = QAEngine()

    # Example usage
    test_chunks = [
        "The Pulsar N160 is a powerful motorcycle with a 160cc engine.",
        "It features advanced fuel injection technology for better performance.",
        "The bike offers excellent mileage of around 45-50 km/l."
    ]

    session_id = "test_session"

    if engine.create_index(test_chunks, session_id):
        answer = engine.answer_question("What is the mileage?", session_id)
        if answer:
            print(f"Answer: {answer}")

        engine.cleanup_session(session_id)
