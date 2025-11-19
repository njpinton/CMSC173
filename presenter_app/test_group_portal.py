"""
Test suite for group portal features including:
- Document upload and retrieval
- Group deletion (admin only)
- File serving
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from io import BytesIO

# Set required environment variables before importing the app
os.environ['FLASK_SECRET_KEY'] = 'test_secret_key_for_testing_only'
os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
os.environ['SUPABASE_ANON_KEY'] = 'test_anon_key'
os.environ['VERCEL_ENV'] = 'development'
os.environ['ADMIN_USERNAME'] = 'testadmin'
from werkzeug.security import generate_password_hash
os.environ['ADMIN_PASSWORD_HASH'] = generate_password_hash('testpassword123')

from api.index import app


@pytest.fixture
def client(monkeypatch):
    """Create Flask test client with testing config."""
    app.config['TESTING'] = True
    # Disable rate limiting for tests
    from api.index import limiter
    monkeypatch.setattr(limiter, 'enabled', False)
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def mock_supabase(monkeypatch):
    """Mock Supabase client."""
    mock_client = MagicMock()
    monkeypatch.setattr('api.index.get_supabase_client', lambda: mock_client)
    return mock_client


@pytest.fixture
def admin_client(client):
    """Create a client with admin session."""
    # Login as admin
    client.post(
        '/admin_login',
        data={'username': 'testadmin', 'password': 'testpassword123'}
    )
    return client


class TestDocumentUpload:
    """Test document upload functionality."""

    def test_upload_document_creates_file(self, client, mock_supabase, monkeypatch, tmp_path):
        """Test that document upload creates a file on disk."""
        # Mock the add_group_document function
        mock_doc = {"id": "doc1", "group_id": "group1", "document_title": "Test Doc", "file_path": str(tmp_path / "test.pdf")}
        monkeypatch.setattr('api.index.add_group_document', lambda *args: mock_doc)

        # Use tmp_path for upload folder
        monkeypatch.setattr('api.index.os.path.dirname', lambda x: str(tmp_path.parent))

        data = {
            'document_title': 'Test Doc',
            'file': (BytesIO(b'PDF content'), 'document.pdf')
        }

        with patch('api.index.allowed_file', return_value=True):
            response = client.post(
                '/api/groups/group1/documents',
                data=data,
                content_type='multipart/form-data'
            )

        assert response.status_code == 201

    def test_upload_document_stores_path_in_db(self, client, mock_supabase, monkeypatch):
        """Test that document upload stores file path in database."""
        captured_args = []

        def mock_add_doc(group_id, title, path):
            captured_args.append((group_id, title, path))
            return {"id": "doc1", "group_id": group_id, "document_title": title, "file_path": path}

        monkeypatch.setattr('api.index.add_group_document', mock_add_doc)

        data = {
            'document_title': 'Test Document',
            'file': (BytesIO(b'PDF content'), 'test.pdf')
        }

        with patch('api.index.allowed_file', return_value=True):
            response = client.post(
                '/api/groups/test-group-id/documents',
                data=data,
                content_type='multipart/form-data'
            )

        assert response.status_code == 201
        assert len(captured_args) == 1
        assert captured_args[0][0] == 'test-group-id'  # group_id
        assert captured_args[0][1] == 'Test Document'  # title
        assert 'test.pdf' in captured_args[0][2]  # file path contains filename


class TestFileServing:
    """Test uploaded file serving."""

    def test_serve_uploaded_file(self, client, tmp_path):
        """Test that uploaded files can be served."""
        # Create a temporary upload folder and file
        upload_folder = tmp_path / "uploads"
        upload_folder.mkdir()
        test_file = upload_folder / "test_file.pdf"
        test_file.write_bytes(b'Test PDF content')

        with patch('api.index.os.path.dirname') as mock_dirname:
            # Mock directory structure
            mock_dirname.return_value = str(tmp_path)

            response = client.get('/uploads/test_file.pdf')

            # File should be served
            assert response.status_code == 200
            assert response.data == b'Test PDF content'

    def test_serve_nonexistent_file(self, client, tmp_path):
        """Test serving a non-existent file returns 404."""
        upload_folder = tmp_path / "uploads"
        upload_folder.mkdir()

        with patch('api.index.os.path.dirname') as mock_dirname:
            mock_dirname.return_value = str(tmp_path)

            response = client.get('/uploads/nonexistent.pdf')
            assert response.status_code == 404

    def test_serve_file_path_traversal_blocked(self, client, tmp_path):
        """Test that path traversal attempts are blocked."""
        # Try to access file outside uploads directory
        response = client.get('/uploads/../../../etc/passwd')

        # Should be blocked (either 400 or 404)
        assert response.status_code in [400, 404]


class TestGroupDeletion:
    """Test group deletion functionality."""

    def test_delete_group_requires_admin(self, client, mock_supabase):
        """Test that group deletion requires admin authentication."""
        response = client.delete('/api/groups/test-group-id')

        # Should return 403 Forbidden without admin auth
        assert response.status_code == 403

    def test_delete_group_as_admin(self, admin_client, mock_supabase, monkeypatch):
        """Test that admin can delete groups."""
        # Mock get_group_details to return a group
        mock_group = {
            'id': 'test-group-id',
            'group_name': 'Test Group',
            'documents': []
        }
        monkeypatch.setattr('api.index.get_group_details', lambda x: mock_group)
        monkeypatch.setattr('api.index.delete_group', lambda x: True)

        response = admin_client.delete('/api/groups/test-group-id')

        assert response.status_code == 200
        data = response.get_json()
        assert 'message' in data
        assert 'deleted' in data['message'].lower()

    def test_delete_group_removes_files(self, admin_client, mock_supabase, monkeypatch, tmp_path):
        """Test that deleting a group removes associated files."""
        # Create test files
        upload_folder = tmp_path / "uploads"
        upload_folder.mkdir()
        test_file1 = upload_folder / "file1.pdf"
        test_file2 = upload_folder / "file2.pdf"
        test_file1.write_bytes(b'File 1 content')
        test_file2.write_bytes(b'File 2 content')

        # Mock group with documents
        mock_group = {
            'id': 'test-group-id',
            'group_name': 'Test Group',
            'documents': [
                {'file_path': str(test_file1), 'document_title': 'File 1'},
                {'file_path': str(test_file2), 'document_title': 'File 2'}
            ]
        }

        monkeypatch.setattr('api.index.get_group_details', lambda x: mock_group)
        monkeypatch.setattr('api.index.delete_group', lambda x: True)

        # Verify files exist before deletion
        assert test_file1.exists()
        assert test_file2.exists()

        response = admin_client.delete('/api/groups/test-group-id')

        assert response.status_code == 200
        # Files should be deleted
        assert not test_file1.exists()
        assert not test_file2.exists()

    def test_delete_nonexistent_group(self, admin_client, mock_supabase, monkeypatch):
        """Test deleting a non-existent group returns 404."""
        monkeypatch.setattr('api.index.get_group_details', lambda x: None)

        response = admin_client.delete('/api/groups/nonexistent-id')

        assert response.status_code == 404

    def test_delete_group_invalid_id(self, admin_client, mock_supabase):
        """Test that invalid group IDs are rejected."""
        # Test with null byte
        response = admin_client.delete('/api/groups/test\x00id')
        assert response.status_code == 400


class TestGroupPortalIntegration:
    """Integration tests for group portal features."""

    def test_upload_and_retrieve_document(self, client, mock_supabase, monkeypatch, tmp_path):
        """Test full workflow: upload document and retrieve it."""
        upload_folder = tmp_path / "uploads"
        upload_folder.mkdir()

        # Mock to use tmp_path
        def mock_dirname(path):
            return str(tmp_path)

        monkeypatch.setattr('api.index.os.path.dirname', mock_dirname)

        # Capture the file path that gets saved
        saved_file_path = None

        def mock_add_doc(group_id, title, path):
            nonlocal saved_file_path
            saved_file_path = path
            return {"id": "doc1", "group_id": group_id, "document_title": title, "file_path": path}

        monkeypatch.setattr('api.index.add_group_document', mock_add_doc)

        # Upload document
        data = {
            'document_title': 'Integration Test Doc',
            'file': (BytesIO(b'Integration test content'), 'integration.pdf')
        }

        with patch('api.index.allowed_file', return_value=True):
            upload_response = client.post(
                '/api/groups/integration-group/documents',
                data=data,
                content_type='multipart/form-data'
            )

        assert upload_response.status_code == 201
        assert saved_file_path is not None

        # File should exist on disk
        assert os.path.exists(saved_file_path)

        # Retrieve the file
        filename = os.path.basename(saved_file_path)
        retrieve_response = client.get(f'/uploads/{filename}')

        assert retrieve_response.status_code == 200
        assert retrieve_response.data == b'Integration test content'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
