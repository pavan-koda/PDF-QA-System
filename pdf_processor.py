import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import re
import io
from PIL import Image

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

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, extract_images: bool = True):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Maximum number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
            extract_images: Whether to extract images from PDFs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images

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

    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images

        Returns:
            List of dictionaries containing image metadata (page, filename, path)
        """
        images_info = []

        if not self.extract_images:
            return images_info

        try:
            if not Path(pdf_path).exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return images_info

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            reader = PdfReader(pdf_path)

            for page_num, page in enumerate(reader.pages):
                try:
                    # Extract images from page
                    if hasattr(page, 'images'):
                        # pypdf method
                        for img_idx, image in enumerate(page.images):
                            try:
                                img_data = image.data
                                img_name = f"page_{page_num + 1}_img_{img_idx + 1}.png"
                                img_path = output_path / img_name

                                # Save image
                                with open(img_path, 'wb') as img_file:
                                    img_file.write(img_data)

                                images_info.append({
                                    'page': page_num + 1,
                                    'filename': img_name,
                                    'path': str(img_path)
                                })

                                logger.info(f"Extracted image: {img_name}")
                            except Exception as e:
                                logger.warning(f"Failed to extract image {img_idx} from page {page_num + 1}: {str(e)}")
                                continue

                    # Alternative: Try XObject extraction (PyPDF2 method)
                    elif '/Resources' in page and '/XObject' in page['/Resources']:
                        xObject = page['/Resources']['/XObject'].get_object()

                        for obj_idx, obj_name in enumerate(xObject):
                            obj = xObject[obj_name]

                            if obj['/Subtype'] == '/Image':
                                try:
                                    size = (obj['/Width'], obj['/Height'])
                                    data = obj.get_data()

                                    # Determine image format
                                    if '/Filter' in obj:
                                        filter_type = obj['/Filter']
                                        if filter_type == '/DCTDecode':
                                            ext = 'jpg'
                                        elif filter_type == '/FlateDecode':
                                            ext = 'png'
                                        elif filter_type == '/JPXDecode':
                                            ext = 'jp2'
                                        else:
                                            ext = 'png'
                                    else:
                                        ext = 'png'

                                    img_name = f"page_{page_num + 1}_img_{obj_idx + 1}.{ext}"
                                    img_path = output_path / img_name

                                    # Try to save as image
                                    if ext == 'jpg' or ext == 'jp2':
                                        with open(img_path, 'wb') as img_file:
                                            img_file.write(data)
                                    else:
                                        # Convert to PNG using PIL
                                        try:
                                            image = Image.frombytes('RGB', size, data)
                                            image.save(img_path, 'PNG')
                                        except:
                                            # Fallback: save raw data
                                            with open(img_path, 'wb') as img_file:
                                                img_file.write(data)

                                    images_info.append({
                                        'page': page_num + 1,
                                        'filename': img_name,
                                        'path': str(img_path)
                                    })

                                    logger.info(f"Extracted image: {img_name}")
                                except Exception as e:
                                    logger.warning(f"Failed to extract XObject image from page {page_num + 1}: {str(e)}")
                                    continue

                except Exception as e:
                    logger.warning(f"Failed to process images on page {page_num + 1}: {str(e)}")
                    continue

            logger.info(f"Successfully extracted {len(images_info)} images from PDF")
            return images_info

        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            return images_info

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
