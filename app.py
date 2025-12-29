from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import os
import pickle
import uuid
import logging
from pathlib import Path
import torch
import numpy as np
import time
from datetime import datetime
from pdf_processor import PDFProcessor
from qa_engine import QAEngine
from config import QA_CONFIG, PDF_CONFIG, FLASK_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = FLASK_CONFIG['max_content_length']
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Initialize QA Engine with config
try:
    from config import EMBEDDING_CONFIG, GENERATOR_CONFIG
except ImportError:
    # Fallback to defaults if config not created yet
    EMBEDDING_CONFIG = {'model_name': 'all-MiniLM-L6-v2'}
    GENERATOR_CONFIG = {'model_name': 'none', 'use_generator': False}

qa_engine = QAEngine(
    embedder_model=EMBEDDING_CONFIG['model_name'],
    gpt2_model=GENERATOR_CONFIG.get('model_name', 'none'),
    use_advanced_qa=QA_CONFIG['use_advanced_qa'],
    advanced_qa_model=QA_CONFIG['advanced_qa_model']
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def log_performance(session_id, question, answer, response_time, model_info):
    """Log performance metrics to a single TXT file."""
    # Use a single performance log file for all sessions
    log_file = Path('logs') / 'performance.txt'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log_entry = f"""
{'='*80}
Session ID: {session_id}
Timestamp: {timestamp}
Question: {question}
Response Time: {response_time:.3f} seconds
Models Used:
  - Embedding: {model_info['embedding']}
  - QA Model: {model_info['qa_model']}
  - Generator: {model_info['generator']}
Answer:
{answer}
{'='*80}

"""

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    return str(log_file)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/qa')
def qa_page():
    if 'session_id' not in session:
        return render_template('upload.html')

    return render_template('qa.html',
                         embedding_model=EMBEDDING_CONFIG.get('model_name', 'Unknown'),
                         qa_model=QA_CONFIG.get('advanced_qa_model', 'Extractive'),
                         generator_model=GENERATOR_CONFIG.get('model_name', 'None'))

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        # Check if file is present
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['pdf_file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        # Save the PDF file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)

        logger.info(f"PDF uploaded: {filepath}")

        # Process the PDF
        # Check if user wants to extract images (default: False)
        extract_images = request.form.get('extract_images', 'false').lower() == 'true'

        pdf_processor = PDFProcessor(
            chunk_size=PDF_CONFIG.get('chunk_size', 400),
            chunk_overlap=PDF_CONFIG.get('chunk_overlap', 50),
            extract_images=extract_images
        )
        text = pdf_processor.extract_text(filepath)

        if not text or len(text.strip()) < 10:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from PDF. The file may be empty or corrupted.'}), 400

        # Split into chunks
        chunks = pdf_processor.split_into_chunks(text)

        if not chunks:
            os.remove(filepath)
            return jsonify({'error': 'Could not create text chunks from PDF.'}), 400

        logger.info(f"Created {len(chunks)} chunks from PDF")

        # Extract images from PDF
        images_dir = os.path.join('images', session_id)
        images_info = pdf_processor.extract_images(filepath, images_dir)
        logger.info(f"Extracted {len(images_info)} images from PDF")

        # Save images metadata with session
        if images_info:
            images_metadata_path = Path('data') / f"{session_id}_images.pkl"
            with open(images_metadata_path, 'wb') as f:
                pickle.dump(images_info, f)

        # Create embeddings and index
        qa_engine.create_index(chunks, session_id)

        # Save metadata
        metadata = {
            'filename': filename,
            'num_chunks': len(chunks),
            'num_images': len(images_info),
            'session_id': session_id
        }

        return jsonify({
            'success': True,
            'message': f'PDF processed successfully! Created {len(chunks)} text chunks.',
            'metadata': metadata
        }), 200

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400

        question = data['question'].strip()

        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        if len(question) > 500:
            return jsonify({'error': 'Question is too long. Maximum 500 characters.'}), 400

        session_id = session.get('session_id')

        if not session_id:
            return jsonify({'error': 'No PDF uploaded. Please upload a PDF first.'}), 400

        # Track response time
        start_time = time.time()

        # Generate answer with full context enabled
        answer = qa_engine.answer_question(
            question,
            session_id,
            use_full_context=QA_CONFIG.get('use_full_context', True)
        )

        # Calculate response time
        response_time = time.time() - start_time

        if not answer:
            return jsonify({'error': 'Could not generate an answer. Please try rephrasing your question.'}), 500

        # Log performance
        model_info = {
            'embedding': EMBEDDING_CONFIG.get('model_name', 'Unknown'),
            'qa_model': QA_CONFIG.get('advanced_qa_model', 'Extractive') if QA_CONFIG.get('use_advanced_qa') else 'Extractive',
            'generator': GENERATOR_CONFIG.get('model_name', 'None')
        }

        log_file = log_performance(session_id, question, answer, response_time, model_info)
        logger.info(f"Logged performance to: {log_file}")

        return jsonify({
            'success': True,
            'answer': answer,
            'question': question,
            'response_time': round(response_time, 3)
        }), 200

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return jsonify({'error': f'Error generating answer: {str(e)}'}), 500

@app.route('/reset', methods=['POST'])
def reset_session():
    try:
        session_id = session.get('session_id')

        if session_id:
            # Clean up files
            qa_engine.cleanup_session(session_id)

            # Clean up uploaded PDF
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                if file.startswith(session_id):
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

        session.clear()

        return jsonify({'success': True, 'message': 'Session reset successfully'}), 200

    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return jsonify({'error': f'Error resetting session: {str(e)}'}), 500

@app.route('/download-log', methods=['GET'])
def download_log():
    try:
        # Download the single performance log file
        log_file = Path('logs') / 'performance.txt'

        if not log_file.exists():
            return jsonify({'error': 'No performance log found'}), 404

        return send_file(
            log_file,
            as_attachment=True,
            download_name=f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mimetype='text/plain'
        )

    except Exception as e:
        logger.error(f"Error downloading log: {str(e)}")
        return jsonify({'error': f'Error downloading log: {str(e)}'}), 500

@app.route('/images', methods=['GET'])
def get_images():
    """Get list of all images extracted from the uploaded PDF"""
    try:
        session_id = session.get('session_id')

        if not session_id:
            return jsonify({'error': 'No PDF uploaded'}), 400

        # Load images metadata
        images_metadata_path = Path('data') / f"{session_id}_images.pkl"

        if not images_metadata_path.exists():
            return jsonify({'images': []}), 200

        with open(images_metadata_path, 'rb') as f:
            images_info = pickle.load(f)

        # Convert absolute paths to relative URLs
        images_list = []
        for img in images_info:
            images_list.append({
                'page': img['page'],
                'filename': img['filename'],
                'url': f"/image/{session_id}/{img['filename']}"
            })

        return jsonify({
            'success': True,
            'images': images_list,
            'total': len(images_list)
        }), 200

    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        return jsonify({'error': f'Error getting images: {str(e)}'}), 500

@app.route('/image/<session_id>/<filename>')
def serve_image(session_id, filename):
    """Serve an extracted image"""
    try:
        # Security check: verify session_id matches current session
        current_session_id = session.get('session_id')
        if current_session_id != session_id:
            return jsonify({'error': 'Unauthorized'}), 403

        image_path = Path('images') / session_id / filename

        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404

        return send_file(image_path, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': f'Error serving image: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': qa_engine.is_ready()
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
