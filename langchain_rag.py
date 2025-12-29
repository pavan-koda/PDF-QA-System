"""
LangChain-based RAG (Retrieval-Augmented Generation) Engine
Provides an alternative to the custom FAISS implementation with LangChain
"""

import logging
from pathlib import Path
from typing import List, Optional
import pickle

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
    import torch
except ImportError as e:
    raise ImportError(f"Missing LangChain dependencies: {str(e)}. Install with: pip install langchain")

logger = logging.getLogger(__name__)


class LangChainRAG:
    """LangChain-based RAG engine for PDF question answering."""

    def __init__(
        self,
        embedder_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "none",
        data_dir: str = "data",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize LangChain RAG engine.

        Args:
            embedder_model: HuggingFace embedding model
            llm_model: HuggingFace LLM model for generation
            data_dir: Directory to store vector stores
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedder_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedder_model,
            model_kwargs={'device': self.device}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize LLM if provided
        self.llm = None
        if llm_model and llm_model.lower() != "none":
            logger.info(f"Loading LLM model: {llm_model}")
            self.llm = self._load_llm(llm_model)

        self.vector_stores = {}  # session_id -> FAISS vector store

    def _load_llm(self, model_id: str):
        """Load LLM using HuggingFace pipeline."""
        try:
            # Detect model path
            model_path = model_id
            local_path = Path(model_id.replace('/', '--'))
            if local_path.exists():
                model_path = str(local_path)

            # Detect model type
            is_seq2seq = 't5' in model_id.lower() or 'flan' in model_id.lower()

            if is_seq2seq:
                # T5/FLAN-T5 models
                hf_pipeline = pipeline(
                    "text2text-generation",
                    model=model_path,
                    tokenizer=model_path,
                    max_new_tokens=512,
                    device=0 if self.device == "cuda" else -1
                )
            else:
                # Causal LM models (GPT-2, Gemma, Llama, etc.)
                hf_pipeline = pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=model_path,
                    max_new_tokens=512,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )

            return HuggingFacePipeline(pipeline=hf_pipeline)

        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            return None

    def create_vector_store(self, text: str, session_id: str) -> bool:
        """
        Create vector store from text.

        Args:
            text: Full text to index
            session_id: Unique session identifier

        Returns:
            True if successful
        """
        try:
            # Split text into chunks
            logger.info("Splitting text into chunks...")
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")

            # Create vector store
            logger.info("Creating vector store...")
            vector_store = FAISS.from_texts(chunks, self.embeddings)

            # Save vector store
            vector_store_path = self.data_dir / f"{session_id}_vectorstore"
            vector_store.save_local(str(vector_store_path))

            # Store in memory
            self.vector_stores[session_id] = vector_store

            logger.info(f"Vector store created for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False

    def load_vector_store(self, session_id: str) -> Optional[FAISS]:
        """Load vector store from disk."""
        try:
            if session_id in self.vector_stores:
                return self.vector_stores[session_id]

            vector_store_path = self.data_dir / f"{session_id}_vectorstore"
            if not vector_store_path.exists():
                logger.error(f"Vector store not found for session: {session_id}")
                return None

            vector_store = FAISS.load_local(
                str(vector_store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.vector_stores[session_id] = vector_store
            return vector_store

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

    def answer_question(
        self,
        question: str,
        session_id: str,
        top_k: int = 3,
        use_llm: bool = False
    ) -> Optional[str]:
        """
        Answer question using RAG.

        Args:
            question: User question
            session_id: Session identifier
            top_k: Number of chunks to retrieve
            use_llm: Whether to use LLM for generation

        Returns:
            Answer string or None
        """
        try:
            # Load vector store
            vector_store = self.load_vector_store(session_id)
            if not vector_store:
                return None

            # Retrieve relevant documents
            docs = vector_store.similarity_search(question, k=top_k)

            if not docs:
                return "No relevant information found."

            # If no LLM or user doesn't want generation, return extractive answer
            if not use_llm or not self.llm:
                return self._extractive_answer(docs, question)

            # Use LLM for generation
            return self._generative_answer(docs, question)

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return None

    def _extractive_answer(self, docs, question: str) -> str:
        """Create extractive answer from retrieved documents."""
        # Combine relevant chunks
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[Chunk {i+1}]\n{doc.page_content}")

        answer = "\n\n".join(context_parts)

        # Truncate if too long
        if len(answer) > 1000:
            answer = answer[:1000] + "\n\n[Answer truncated...]"

        return answer

    def _generative_answer(self, docs, question: str) -> str:
        """Generate answer using LLM."""
        try:
            # Create retriever
            retriever = self.vector_stores[list(self.vector_stores.keys())[0]].as_retriever(
                search_kwargs={"k": 3}
            )

            # Create prompt template
            prompt_template = """Use the following context to answer the question. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False
            )

            # Get answer
            result = qa_chain({"query": question})
            return result['result']

        except Exception as e:
            logger.error(f"Error in generative answer: {str(e)}")
            # Fallback to extractive
            return self._extractive_answer(docs, question)

    def cleanup_session(self, session_id: str) -> bool:
        """Clean up session data."""
        try:
            # Remove from memory
            if session_id in self.vector_stores:
                del self.vector_stores[session_id]

            # Remove from disk
            vector_store_path = self.data_dir / f"{session_id}_vectorstore"
            if vector_store_path.exists():
                import shutil
                shutil.rmtree(vector_store_path)

            logger.info(f"Cleaned up session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up session: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the LangChain RAG engine
    logging.basicConfig(level=logging.INFO)

    rag = LangChainRAG(
        embedder_model="all-MiniLM-L6-v2",
        llm_model="none"
    )

    # Test text
    test_text = """
    The Pulsar N160 is a powerful motorcycle with a 160cc engine.
    It features advanced fuel injection technology for better performance.
    The bike offers excellent mileage of around 45-50 km/l.
    The price starts from Rs. 1,20,000 for the base variant.
    """

    session_id = "test_langchain"

    if rag.create_vector_store(test_text, session_id):
        answer = rag.answer_question("What is the mileage?", session_id)
        if answer:
            print(f"Answer:\n{answer}")

        rag.cleanup_session(session_id)
