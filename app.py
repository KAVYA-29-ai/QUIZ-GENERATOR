from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import requests
import PyPDF2
import io
import re
from typing import Dict, List, Any

###Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Bubble.io Integration Configuration
# Make sure to set these as environment variables in Railway
BUBBLE_APP_BASE_URL = os.environ.get(
    "BUBBLE_APP_BASE_URL",
    "https://xxerror707-94447.bubbleapps.io/version-test"
).rstrip("/")
BUBBLE_API_TOKEN = os.environ.get("BUBBLE_API_TOKEN")
# For version-test (dev) use "/version-test/api/1.1/obj", for live omit version-test
BUBBLE_API_VERSION_PATH = "/version-test/api/1.1/obj"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using PyPDF2"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts: List[str] = []

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue

        text = "".join(text_parts)
        # Cleanup: collapse excessive whitespace but keep paragraphs
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +\n', '\n', text)
        return text.strip()

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""


async def generate_questions_with_claude(text_content: str, num_mcq: int = 5, num_tf: int = 5) -> Dict[str, Any]:
    """Generate questions using Claude API."""
    # Shorten content if too long
    if len(text_content) > 8000:
        text_content = text_content[:8000] + "...[Content truncated for processing]"

    prompt = f"""Based on the following textbook chapter content, generate educational questions for students:

CONTENT:
{text_content}

Please generate exactly {num_mcq} multiple choice questions and {num_tf} true/false questions based on this content.

Your response must be a valid JSON object with this exact structure:
{{
    "multiple_choice": [
        {{
            "question": "Question text here?",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correct_answer": "A",
            "explanation": "Brief explanation of why this is correct"
        }}
    ],
    "true_false": [
        {{
            "question": "Statement to evaluate as true or false",
            "correct_answer": true,
            "explanation": "Brief explanation of why this is true/false"
        }}
    ]
}}

Guidelines:
1. Questions should test understanding, not just memorization
2. Multiple choice options should be plausible but clearly distinguishable
3. True/false questions should be clear statements that can be definitively evaluated
4. Include brief explanations for educational value
5. Focus on key concepts from the provided content
6. Make questions appropriate for the academic level of the content

RESPOND ONLY WITH VALID JSON. DO NOT INCLUDE ANY TEXT OUTSIDE THE JSON STRUCTURE."""

    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("Anthropic API key not set in environment variables.")
            return generate_fallback_questions()

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key
            },
            json={
                "model": "claude-3-5-sonnet-20240620",  # Latest stable Claude model
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Claude API error: {response.status_code} - {response.text}")
            return generate_fallback_questions()

        response_data = response.json()
        claude_text = ""

        if isinstance(response_data, dict):
            if 'content' in response_data and isinstance(response_data['content'], list) and len(response_data['content']) > 0:
                claude_text = response_data['content'][0].get('text', '')
            else:
                claude_text = response_data.get('text', '')
        else:
            claude_text = str(response_data)

        # Clean JSON text
        claude_text = (claude_text or "").strip()
        if claude_text.startswith("```json"):
            claude_text = claude_text[7:]
        if claude_text.endswith("```"):
            claude_text = claude_text[:-3]
        claude_text = claude_text.strip()

        questions_data = json.loads(claude_text)
        if not isinstance(questions_data, dict):
            raise ValueError("Claude response is not a JSON object")
        if 'multiple_choice' not in questions_data or 'true_false' not in questions_data:
            raise ValueError("Response missing required fields")

        return questions_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Claude API: {str(e)}")
        return generate_fallback_questions()
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing Claude response as JSON: {str(e)}")
        return generate_fallback_questions()
    except Exception as e:
        logger.error(f"Unexpected error generating questions: {str(e)}")
        return generate_fallback_questions()


def generate_fallback_questions():
    """Fallback questions if LLM fails"""
    return {
        "multiple_choice": [
            {
                "question": "Based on the uploaded content, which concept appears to be most important?",
                "options": [
                    "A) The first concept mentioned",
                    "B) The most frequently discussed topic",
                    "C) The conclusion of the chapter",
                    "D) All concepts are equally important"
                ],
                "correct_answer": "B",
                "explanation": "The most frequently discussed topics usually indicate key concepts in educational content."
            }
        ],
        "true_false": [
            {
                "question": "The uploaded document contains educational content suitable for generating practice questions.",
                "correct_answer": True,
                "explanation": "Since the document was processed successfully, it contains readable educational content."
            }
        ]
    }


def post_to_bubble(data_type: str, payload: dict) -> dict:
    """Post data to Bubble.io Data API (object create). Returns parsed JSON or error dict."""
    if not BUBBLE_API_TOKEN:
        logger.warning("Bubble API token not configured; skipping Bubble integration")
        return {"skipped": True, "reason": "No API token"}

    url = f"{BUBBLE_APP_BASE_URL}{BUBBLE_API_VERSION_PATH}/{data_type}"
    headers = {
        "Authorization": f"Bearer {BUBBLE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Bubble POST error for {data_type}: {e} - response: {getattr(e, 'response', None)}")
        # try return response body if available for debugging
        try:
            return {"error": str(e), "response_text": e.response.text}
        except Exception:
            return {"error": str(e)}


def push_questions_to_bubble(questions_data: dict, file_id: str, original_filename: str) -> dict:
    """Push generated questions to Bubble.io database"""
    results = {
        "quiz_created": False,
        "mcq_created": 0,
        "tf_created": 0,
        "errors": []
    }

    try:
        # Create Quiz object (optional)
        quiz_payload = {
            "title": f"Quiz for {original_filename}",
            "source_file_id": file_id,
            "created_at": datetime.now().isoformat(),
            "total_mcq": len(questions_data.get("multiple_choice", [])),
            "total_tf": len(questions_data.get("true_false", []))
        }
        quiz_resp = post_to_bubble("Quiz", quiz_payload)
        bubble_quiz_id = None

        if isinstance(quiz_resp, dict) and "id" in quiz_resp:
            bubble_quiz_id = quiz_resp["id"]
            results["quiz_created"] = True
            logger.info(f"Created Quiz in Bubble with ID: {bubble_quiz_id}")
        elif quiz_resp.get("skipped"):
            logger.info("Bubble integration skipped - no API token")
            return {"skipped": True}
        else:
            results["errors"].append(f"Failed to create Quiz: {quiz_resp}")

        # Push MCQs
        for i, mcq in enumerate(questions_data.get("multiple_choice", [])):
            try:
                options = mcq.get("options", ["", "", "", ""])
                payload = {
                    "question_text": mcq.get("question", ""),
                    "option_A": options[0] if len(options) > 0 else "",
                    "option_B": options[1] if len(options) > 1 else "",
                    "option_C": options[2] if len(options) > 2 else "",
                    "option_D": options[3] if len(options) > 3 else "",
                    "correct_option": mcq.get("correct_answer", "A"),
                    "explanation": mcq.get("explanation", ""),
                    "source_file_id": file_id,
                    "question_type": "multiple_choice",
                    "question_number": i + 1
                }

                if bubble_quiz_id:
                    payload["quiz"] = bubble_quiz_id

                mcq_resp = post_to_bubble("Question", payload)
                if isinstance(mcq_resp, dict) and "id" in mcq_resp:
                    results["mcq_created"] += 1
                else:
                    results["errors"].append(f"Failed to create MCQ {i+1}: {mcq_resp}")

            except Exception as e:
                results["errors"].append(f"Error processing MCQ {i+1}: {str(e)}")

        # Push True/False questions
        for i, tf in enumerate(questions_data.get("true_false", [])):
            try:
                payload = {
                    "statement": tf.get("question", ""),
                    "answer_tf": bool(tf.get("correct_answer", False)),
                    "explanation": tf.get("explanation", ""),
                    "source_file_id": file_id,
                    "question_type": "true_false",
                    "question_number": i + 1
                }

                if bubble_quiz_id:
                    payload["quiz"] = bubble_quiz_id

                tf_resp = post_to_bubble("TFQuestion", payload)
                if isinstance(tf_resp, dict) and "id" in tf_resp:
                    results["tf_created"] += 1
                else:
                    results["errors"].append(f"Failed to create T/F {i+1}: {tf_resp}")

            except Exception as e:
                results["errors"].append(f"Error processing T/F {i+1}: {str(e)}")

        logger.info(f"Bubble integration results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error in push_questions_to_bubble: {str(e)}")
        return {"error": str(e)}


@app.route('/')
def index():
    return jsonify({
        "message": "QuizGenius Backend API with Bubble.io Integration",
        "version": "1.1.0",
        "endpoints": {
            "upload": "/upload",
            "generate": "/generate",
            "generate_and_push": "/generate-and-push",
            "health": "/health"
        },
        "bubble_integration": {
            "configured": bool(BUBBLE_API_TOKEN),
            "base_url": BUBBLE_APP_BASE_URL
        }
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "QuizGenius Backend",
        "bubble_integration": bool(BUBBLE_API_TOKEN)
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Accepts multipart/form-data with key "file" (PDF).
    Saves PDF, extracts text, stores text to outputs/<file_id>_content.txt and returns a file_id.
    This is the endpoint Bubble should call with Send file = true.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided (key "file" missing)'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Generate unique filename and save
        file_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        filename = f"{file_id}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)
        logger.info(f"Saved uploaded file to {filepath}")

        # Extract text
        with open(filepath, 'rb') as pdf_file:
            text_content = extract_text_from_pdf(pdf_file)

        if not text_content.strip():
            os.remove(filepath)
            logger.warning("Text extraction returned empty")
            return jsonify({'error': 'Could not extract text from PDF. Ensure the PDF contains selectable text.'}), 400

        # Store file info
        file_info = {
            'file_id': file_id,
            'original_filename': file.filename,
            'stored_filename': filename,
            'upload_time': datetime.now().isoformat(),
            'text_length': len(text_content),
            'text_preview': text_content[:200] + "..." if len(text_content) > 200 else text_content
        }

        # Save extracted text for later question generation
        text_file_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_content.txt")
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)

        logger.info(f"Extracted text saved to {text_file_path}")

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'file_info': file_info,
            'text_extracted': True
        }), 200

    except Exception as e:
        logger.exception("Error in upload endpoint")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/generate', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        file_id = data.get('file_id')
        num_mcq = data.get('num_mcq', 5)
        num_tf = data.get('num_tf', 5)

        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400

        # Validate question counts
        if not isinstance(num_mcq, int) or not isinstance(num_tf, int):
            return jsonify({'error': 'num_mcq and num_tf must be integers'}), 400

        if num_mcq < 1 or num_mcq > 20 or num_tf < 1 or num_tf > 20:
            return jsonify({'error': 'Question counts must be between 1 and 20'}), 400

        # Read extracted text
        text_file_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_content.txt")
        if not os.path.exists(text_file_path):
            return jsonify({'error': 'File not found. Please upload the file first.'}), 404

        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        if not text_content.strip():
            return jsonify({'error': 'No content found in the uploaded file'}), 400

        logger.info(f"Generating {num_mcq} MCQ and {num_tf} T/F for file {file_id}")

        # Generate questions (synchronous wrapper)
        import asyncio
        questions_data = asyncio.run(generate_questions_with_claude(text_content, num_mcq, num_tf))

        # Save generated questions
        output_data = {
            'file_id': file_id,
            'generation_time': datetime.now().isoformat(),
            'questions': questions_data,
            'settings': {
                'num_mcq': num_mcq,
                'num_tf': num_tf
            }
        }

        output_file_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_questions.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return jsonify({
            'message': 'Questions generated successfully',
            'file_id': file_id,
            'questions': questions_data,
            'total_questions': len(questions_data.get('multiple_choice', [])) + len(questions_data.get('true_false', []))
        }), 200

    except Exception as e:
        logger.exception("Error in generate endpoint")
        return jsonify({'error': f'Question generation failed: {str(e)}'}), 500


@app.route('/generate-and-push', methods=['POST'])
def generate_and_push_to_bubble():
    """Generate questions and push them to Bubble DB objects."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        file_id = data.get('file_id')
        num_mcq = data.get('num_mcq', 5)
        num_tf = data.get('num_tf', 5)

        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400

        # Read extracted text
        text_file_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_content.txt")
        if not os.path.exists(text_file_path):
            return jsonify({'error': 'File not found. Please upload the file first.'}), 404

        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        if not text_content.strip():
            return jsonify({'error': 'No content found in the uploaded file'}), 400

        # Get original filename for Quiz title
        upload_files = os.listdir(UPLOAD_FOLDER)
        original_filename = "Unknown File"
        for upload_file in upload_files:
            if upload_file.startswith(file_id):
                original_filename = upload_file.split('_', 1)[1] if '_' in upload_file else upload_file
                break

        logger.info(f"Generating and pushing {num_mcq} MCQ and {num_tf} T/F for file {file_id}")

        import asyncio
        questions_data = asyncio.run(generate_questions_with_claude(text_content, num_mcq, num_tf))

        # Push to Bubble
        bubble_results = push_questions_to_bubble(questions_data, file_id, original_filename)

        # Save locally
        output_data = {
            'file_id': file_id,
            'generation_time': datetime.now().isoformat(),
            'questions': questions_data,
            'settings': {
                'num_mcq': num_mcq,
                'num_tf': num_tf
            },
            'bubble_integration': bubble_results
        }

        output_file_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_questions.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return jsonify({
            'message': 'Questions generated and pushed to Bubble successfully',
            'file_id': file_id,
            'questions': questions_data,
            'total_questions': len(questions_data.get('multiple_choice', [])) + len(questions_data.get('true_false', [])),
            'bubble_integration': bubble_results
        }), 200

    except Exception as e:
        logger.exception("Error in generate-and-push endpoint")
        return jsonify({'error': f'Question generation and push failed: {str(e)}'}), 500


@app.route('/download/<file_id>')
def download_questions(file_id):
    try:
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_questions.json")
        if not os.path.exists(output_file_path):
            return jsonify({'error': 'Questions file not found'}), 404

        return send_from_directory(OUTPUT_FOLDER, f"{file_id}_questions.json",
                                   as_attachment=True,
                                   download_name=f"quiz_questions_{file_id}.json")

    except Exception as e:
        logger.exception("Error in download endpoint")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/files/<file_id>')
def get_file_info(file_id):
    try:
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_questions.json")
        if not os.path.exists(output_file_path):
            return jsonify({'error': 'File not found'}), 404

        with open(output_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return jsonify(data), 200

    except Exception as e:
        logger.exception("Error in get_file_info endpoint")
        return jsonify({'error': f'Failed to get file info: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting QuizGenius Backend with Bubble.io Integration on port {port}")
    logger.info(f"Bubble integration configured: {bool(BUBBLE_API_TOKEN)}")
    app.run(host='0.0.0.0', port=port, debug=debug)
