import logging
from pathlib import Path
from typing import List, Optional
import re

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("Please install pypdf or PyPDF2: pip install pypdf")

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and chunking with robust error handling."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Maximum number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            if not Path(pdf_path).exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None

            reader = PdfReader(pdf_path)
            text = ""

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                    continue

            if not text.strip():
                logger.error("No text could be extracted from the PDF")
                return None

            # Clean up the text
            text = self._clean_text(text)

            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and special characters.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers/footers (basic cleanup)
        text = re.sub(r'\n\d+\n', '\n', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)

        return text.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        try:
            if not text or not text.strip():
                logger.error("Cannot split empty text into chunks")
                return []

            # Split into sentences first
            sentences = self._split_into_sentences(text)

            if not sentences:
                logger.warning("No sentences found in text")
                return [text]

            chunks = []
            current_chunk = []
            current_word_count = 0

            for sentence in sentences:
                sentence_words = sentence.split()
                sentence_word_count = len(sentence_words)

                # If adding this sentence would exceed chunk size
                if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    overlap_words = []
                    overlap_count = 0

                    # Add words from end of previous chunk for overlap
                    for sent in reversed(current_chunk):
                        sent_words = sent.split()
                        if overlap_count + len(sent_words) <= self.chunk_overlap:
                            overlap_words.insert(0, sent)
                            overlap_count += len(sent_words)
                        else:
                            break

                    current_chunk = overlap_words
                    current_word_count = overlap_count

                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_word_count += sentence_word_count

            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)

            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error splitting text into chunks: {str(e)}")
            return []

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitter using regex
        # Splits on . ! ? followed by space and capital letter
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def save_text(self, text: str, output_path: str) -> bool:
        """
        Save extracted text to a file.

        Args:
            text: Text to save
            output_path: Path to save the text

        Returns:
            True if successful, False otherwise
        """
        try:
            Path(output_path).write_text(text, encoding='utf-8')
            logger.info(f"Saved text to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving text to file: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the processor
    logging.basicConfig(level=logging.INFO)

    processor = PDFProcessor()

    # Example usage
    pdf_path = "test.pdf"
    if Path(pdf_path).exists():
        text = processor.extract_text(pdf_path)
        if text:
            chunks = processor.split_into_chunks(text)
            print(f"Extracted {len(chunks)} chunks from PDF")
            print(f"First chunk preview: {chunks[0][:200]}...")
    else:
        print(f"Test PDF not found: {pdf_path}")
