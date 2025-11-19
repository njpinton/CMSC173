from flask import Flask, render_template, send_from_directory, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import re
import logging
from supabase import create_client, Client
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
import secrets
from supabase_client import get_supabase_client, create_group, add_group_member, add_group_document, get_groups, get_group_details, delete_group

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -- Supabase Configuration --
# To run this locally, you will need to create a .env file in the presenter_app directory
# with the following content:
# SUPABASE_URL=your_supabase_url
# SUPABASE_ANON_KEY=your_supabase_anon_key
# FLASK_SECRET_KEY=your_secret_key (required - no default fallback)
# ADMIN_USERNAME=admin
# ADMIN_PASSWORD_HASH=hashed_password (use: python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password'))")
#
# You will also need to create a table in your Supabase project with the following schema:
# CREATE TABLE module_views (
#   module_number INT PRIMARY KEY,
#   view_count INT
# );
#
# And a function to increment the view count:
# CREATE OR REPLACE FUNCTION increment_module_view(module_id INT)
# RETURNS VOID AS $$
# BEGIN
#   INSERT INTO module_views (module_number, view_count)
#   VALUES (module_id, 1)
#   ON CONFLICT (module_number)
#   DO UPDATE SET view_count = module_views.view_count + 1;
# END;
# $$ LANGUAGE plpgsql;
#
# To get your Supabase URL and anon key, go to your Supabase project's
# API settings page.

# The global supabase client initialization is now handled in supabase_client.py
# --------------------------

app = Flask(__name__)

# Session configuration
from datetime import timedelta
from functools import wraps
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # 2 hour session timeout
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('VERCEL_ENV') == 'production'  # HTTPS only in production
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

# Secret key management - secure by default
secret_key = os.environ.get('FLASK_SECRET_KEY')
if not secret_key:
    # In production (Vercel), use a derived key from Vercel's system
    # In development, warn but allow with a generated key
    is_production = os.environ.get('VERCEL_ENV') == 'production'

    if is_production:
        # For Vercel production, derive a key from existing secrets
        vercel_url = os.environ.get('VERCEL_URL', 'localhost')
        secret_key = f"vercel-{vercel_url}-{os.environ.get('VERCEL_GIT_COMMIT_SHA', 'dev')[:16]}"
        logger.warning("Using derived FLASK_SECRET_KEY from Vercel environment. Set FLASK_SECRET_KEY for better control.")
    else:
        # For local development, generate a key
        import secrets
        secret_key = secrets.token_urlsafe(32)
        logger.warning("Generated temporary FLASK_SECRET_KEY for development. Set FLASK_SECRET_KEY environment variable for production.")

app.secret_key = secret_key

# Configure CORS - only allow specified origins in production
allowed_origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:*').split(',')
CORS(app, resources={
    r"/api/*": {"origins": allowed_origins, "methods": ["GET", "POST", "OPTIONS"]},
}, supports_credentials=True)
logger.info(f"CORS configured for origins: {allowed_origins}")

# Rate limiting configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Use Redis in production for distributed systems
)
logger.info("Rate limiting configured")

# Security configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_MIME_TYPES = {
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'csv', 'xls', 'xlsx'}

def allowed_file(filename: str, mime_type: str = None) -> bool:
    """Validate file type against allowed extensions and MIME types."""
    if not filename or '.' not in filename:
        return False

    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False

    # Additional MIME type check if provided
    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        return False

    return True

def validate_input(value: str, max_length: int = 255, field_name: str = "input") -> tuple[bool, str]:
    """Validate string input for length and null bytes."""
    if not isinstance(value, str):
        return False, f"{field_name} must be a string"

    if not value or len(value.strip()) == 0:
        return False, f"{field_name} cannot be empty"

    if len(value) > max_length:
        return False, f"{field_name} exceeds maximum length of {max_length}"

    if '\x00' in value:
        return False, f"{field_name} contains invalid characters"

    return True, ""

def admin_required(f):
    """Decorator to require admin authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or not session.get('is_admin'):
            logger.warning(f"Unauthorized access attempt to {request.endpoint}")
            return jsonify({"error": "Unauthorized - Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated_function

# Module names and filenames for display
MODULES = {
    0: {
        "title": "Introduction to Machine Learning",
        "filename": "00-intro-to-machine-learning.html"
    },
    1: {
        "title": "Parameter Estimation",
        "filename": "01-parameter-estimation.html"
    },
    2: {
        "title": "Linear Regression",
        "filename": "02-linear-regression.html"
    },
    3: {
        "title": "Regularization",
        "filename": "03-regularization.html"
    },
    4: {
        "title": "Exploratory Data Analysis",
        "filename": "04-exploratory-data-analysis.html"
    },
    5: {
        "title": "Model Selection",
        "filename": "05-model-selection.html"
    },
    6: {
        "title": "Cross Validation",
        "filename": "06-cross-validation.html"
    },
    7: {
        "title": "PCA",
        "filename": "07-pca.html"
    },
    8: {
        "title": "Logistic Regression",
        "filename": "08-logistic-regression.html"
    },
    9: {
        "title": "Classification",
        "filename": "09-classification.html"
    },
    10: {
        "title": "Kernel Methods",
        "filename": "10-kernel-methods.html"
    },
    11: {
        "title": "Clustering",
        "filename": "11-clustering.html"
    },
    12: {
        "title": "Neural Networks",
        "filename": "12-neural-networks.html"
    },
    13: {
        "title": "Advanced Neural Networks",
        "filename": "13-advanced-neural-networks.html"
    },
}

def get_available_modules():
    """Dynamically discover available module templates"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    available = {}

    for module_num, module_info in MODULES.items():
        filename = module_info['filename']
        filepath = os.path.join(templates_dir, filename)

        if os.path.exists(filepath):
            available[module_num] = module_info

    return available

@app.route('/')
def index():
    available_modules = get_available_modules()
    return render_template('index.html', modules=available_modules)

@app.route('/module/<int:module_number>')
def show_module(module_number):
    available_modules = get_available_modules()

    if module_number not in available_modules:
        return f"""
        <html>
        <head>
            <title>Module Not Found</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 40px; background: #f0f0f0; }}
                .error {{ background: white; padding: 20px; border-radius: 8px; }}
                h1 {{ color: #d32f2f; }}
                a {{ color: #667eea; text-decoration: none; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h1>Module {module_number} Not Found</h1>
                <p>Available modules: {sorted(available_modules.keys())}</p>
                <p><a href="/">← Back to homepage</a></p>
                </div>
            </body>
            </html>
            """, 404

    # -- Supabase Integration --
    view_count = 0
    supabase_client = get_supabase_client()
    if supabase_client:
        try:
            # Increment view count
            supabase_client.rpc('increment_module_view', {'module_id': module_number}).execute()

            # Get view count
            response = supabase_client.table('module_views').select('view_count').eq('module_number', module_number).execute()
            if response.data:
                view_count = response.data[0]['view_count']
            logger.info(f"Module {module_number} view count: {view_count}")
        except Exception as e:
            logger.error(f"Error interacting with Supabase for module {module_number}: {e}", exc_info=True)
            view_count = 0  # Default to 0 instead of "Error" string
    # --------------------------

    module = available_modules[module_number]
    return render_template(module['filename'], view_count=view_count)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve images from the images directory"""
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    return send_from_directory(images_dir, filename)

@app.route('/api/groups', methods=['POST'])
@limiter.limit("20 per hour")  # Rate limit group creation
def create_group_api():
    supabase_client = get_supabase_client()
    if not supabase_client:
        logger.error("Supabase client not configured")
        return jsonify({"error": "Supabase not configured"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        group_name = data.get('group_name', '').strip()
        project_title = data.get('project_title', '').strip()
        members = data.get('members', [])

        # Validate group_name
        is_valid, error_msg = validate_input(group_name, 100, "group_name")
        if not is_valid:
            logger.warning(f"Invalid group_name: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Validate project_title if provided
        if project_title:
            is_valid, error_msg = validate_input(project_title, 255, "project_title")
            if not is_valid:
                logger.warning(f"Invalid project_title: {error_msg}")
                return jsonify({"error": error_msg}), 400

        # Validate members list
        if not isinstance(members, list):
            return jsonify({"error": "members must be a list"}), 400

        if len(members) > 50:  # Reasonable limit
            return jsonify({"error": "Too many members (max 50)"}), 400

        for member in members:
            is_valid, error_msg = validate_input(member, 100, "member_name")
            if not is_valid:
                logger.warning(f"Invalid member: {error_msg}")
                return jsonify({"error": f"Invalid member name: {error_msg}"}), 400

        new_group = create_group(group_name, project_title)
        if new_group:
            group_id = new_group['id']
            logger.info(f"Created group {group_id} with name '{group_name}'")
            for member_name in members:
                add_group_member(group_id, member_name)
            return jsonify(new_group), 201
        return jsonify({"error": "Failed to create group"}), 500
    except Exception as e:
        logger.error(f"Error creating group: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/groups', methods=['GET'])
def get_groups_api():
    supabase_client = get_supabase_client()
    if not supabase_client:
        return jsonify({"error": "Supabase not configured"}), 500
    try:
        groups = get_groups()
        return jsonify(groups), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/groups/<group_id>', methods=['GET'])
def get_group_details_api(group_id):
    supabase_client = get_supabase_client()
    if not supabase_client:
        return jsonify({"error": "Supabase not configured"}), 500
    try:
        group_details = get_group_details(group_id)
        if group_details:
            return jsonify(group_details), 200
        return jsonify({"error": "Group not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/groups/<group_id>/documents', methods=['POST'])
@limiter.limit("10 per hour")  # Rate limit file uploads
def upload_document_api(group_id):
    supabase_client = get_supabase_client()
    if not supabase_client:
        logger.error("Supabase client not configured for document upload")
        return jsonify({"error": "Supabase not configured"}), 500

    try:
        # Validate group_id format
        is_valid, error_msg = validate_input(group_id, 255, "group_id")
        if not is_valid:
            logger.warning(f"Invalid group_id: {error_msg}")
            return jsonify({"error": error_msg}), 400

        if 'file' not in request.files:
            logger.warning("File upload attempt with no file part")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Validate file size
        if request.content_length and request.content_length > MAX_FILE_SIZE:
            logger.warning(f"File size exceeds maximum: {request.content_length} > {MAX_FILE_SIZE}")
            return jsonify({"error": f"File size exceeds maximum of {MAX_FILE_SIZE // (1024*1024)} MB"}), 413

        # Validate file type
        if not allowed_file(file.filename, file.content_type):
            logger.warning(f"File type not allowed: {file.filename} ({file.content_type})")
            return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        document_title = request.form.get('document_title', '').strip() or file.filename

        # Validate document title
        is_valid, error_msg = validate_input(document_title, 255, "document_title")
        if not is_valid:
            logger.warning(f"Invalid document_title: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Store files locally in uploads directory
        # TODO: Consider migrating to cloud storage (Supabase Storage, S3)
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        # Use secure filename and add timestamp to prevent collisions
        import time
        secure_name = secure_filename(file.filename)
        filename = f"{int(time.time())}_{secure_name}"
        file_path = os.path.join(upload_folder, filename)

        # Ensure file_path is within upload_folder (path traversal prevention)
        real_path = os.path.realpath(file_path)
        real_upload_folder = os.path.realpath(upload_folder)
        if not real_path.startswith(real_upload_folder):
            logger.error(f"Path traversal attempt detected: {file_path}")
            return jsonify({"error": "Invalid file path"}), 400

        file.save(file_path)
        logger.info(f"File saved: {filename} for group {group_id}")

        try:
            new_document = add_group_document(group_id, document_title, file_path)
            if new_document:
                logger.info(f"Document metadata added for group {group_id}")
                return jsonify(new_document), 201
            return jsonify({"error": "Failed to add document metadata"}), 500
        except Exception as e:
            logger.error(f"Error adding document metadata: {e}", exc_info=True)
            # Clean up file if metadata insertion fails
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({"error": "Failed to process document"}), 500
    except Exception as e:
        logger.error(f"Error in file upload: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/group_portal')
def group_portal():
    is_admin = session.get('logged_in', False) and session.get('is_admin', False)
    return render_template('group_portal.html', is_admin=is_admin)

@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    """Admin-only dashboard showing all groups and their submissions."""
    return render_template('admin_dashboard.html')

@app.route('/admin_login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # Strict rate limit for login attempts
def admin_login():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')

            # Validate input
            if not username or not password:
                logger.warning("Login attempt with empty credentials")
                return render_template('admin_login.html', error='Username and password are required')

            admin_username = os.environ.get('ADMIN_USERNAME', '')
            admin_password_hash = os.environ.get('ADMIN_PASSWORD_HASH', '')

            if not admin_username or not admin_password_hash:
                logger.error("Admin credentials not properly configured")
                return render_template('admin_login.html', error='Server misconfiguration')

            # Use constant-time comparison to prevent timing attacks
            username_match = secrets.compare_digest(username, admin_username)
            password_match = check_password_hash(admin_password_hash, password)

            if username_match and password_match:
                session['logged_in'] = True
                session['is_admin'] = True
                session.permanent = True  # Enable session timeout
                logger.info(f"Admin login successful for user {username}")
                return redirect(url_for('group_portal'))
            else:
                # Use generic message to prevent user enumeration
                logger.warning(f"Failed login attempt for user {username}")
                return render_template('admin_login.html', error='Invalid credentials')
        except Exception as e:
            logger.error(f"Error in admin login: {e}", exc_info=True)
            return render_template('admin_login.html', error='An error occurred during login')
    return render_template('admin_login.html')

@app.route('/admin_logout', methods=['GET', 'POST'])
def admin_logout():
    """Logout admin user and clear session."""
    if session.get('logged_in'):
        username = session.get('username', 'unknown')
        logger.info(f"Admin logout for user {username}")
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/groups/<group_id>', methods=['DELETE'])
@admin_required
def delete_group_api(group_id):
    """Delete a group and all associated data (admin only)."""
    supabase_client = get_supabase_client()
    if not supabase_client:
        return jsonify({"error": "Supabase not configured"}), 500

    try:
        # Validate group_id
        is_valid, error_msg = validate_input(group_id, 255, "group_id")
        if not is_valid:
            logger.warning(f"Invalid group_id for deletion: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Get group details to find associated files
        group_details = get_group_details(group_id)
        if not group_details:
            return jsonify({"error": "Group not found"}), 404

        # Delete physical files
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        if group_details.get('documents'):
            for doc in group_details['documents']:
                file_path = doc.get('file_path', '')
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")

        # Delete group from database
        success = delete_group(group_id)
        if success:
            logger.info(f"Group {group_id} deleted by admin")
            return jsonify({"message": "Group deleted successfully"}), 200
        return jsonify({"error": "Failed to delete group"}), 500
    except Exception as e:
        logger.error(f"Error in delete_group_api: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """Serve uploaded files from the uploads directory."""
    # Security: Use secure_filename to prevent directory traversal
    safe_filename = secure_filename(filename)
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')

    # Verify file exists and is within upload folder
    file_path = os.path.join(upload_folder, safe_filename)
    real_path = os.path.realpath(file_path)
    real_upload_folder = os.path.realpath(upload_folder)

    if not real_path.startswith(real_upload_folder):
        logger.warning(f"Path traversal attempt detected for file: {filename}")
        return jsonify({"error": "Invalid file path"}), 400

    if not os.path.exists(file_path):
        logger.warning(f"File not found: {filename}")
        return jsonify({"error": "File not found"}), 404

    try:
        return send_from_directory(upload_folder, safe_filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}", exc_info=True)
        return jsonify({"error": "Error serving file"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8788)