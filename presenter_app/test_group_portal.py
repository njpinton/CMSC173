"""
Test suite for group portal features including:
- Document upload and retrieval (Supabase Storage)
- Group deletion (admin only)
- File serving via Supabase Storage URLs
"""

import pytest
import os
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
    """Test document upload functionality to Supabase Storage."""

    def test_upload_document_to_storage(self, client, mock_supabase, monkeypatch):
        """Test that document upload uses Supabase Storage."""
        # Mock storage helper
        def mock_upload(file, group_id, document_title):
            return (f"group-documents/{group_id}/test_file.pdf", 1024, "application/pdf")

        monkeypatch.setattr('api.index.upload_file_to_storage', mock_upload)

        # Mock the add_group_document function
        mock_doc = {
            "id": "doc1",
            "group_id": "group1",
            "document_title": "Test Doc",
            "file_path": "group-documents/group1/test_file.pdf"
        }
        monkeypatch.setattr('api.index.add_group_document', lambda *args: mock_doc)

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
        json_data = response.get_json()
        # Response is the document object directly
        assert json_data['file_path'] == 'group-documents/group1/test_file.pdf'
        assert json_data['file_size'] == 1024
        assert json_data['mime_type'] == 'application/pdf'

    def test_upload_document_stores_storage_path_in_db(self, client, mock_supabase, monkeypatch):
        """Test that document upload stores Supabase Storage path in database."""
        # Mock storage helper
        def mock_upload(file, group_id, document_title):
            return (f"group-documents/{group_id}/test_file.pdf", 1024, "application/pdf")

        monkeypatch.setattr('api.index.upload_file_to_storage', mock_upload)

        captured_args = []

        def mock_add_doc(group_id, title, path):
            captured_args.append((group_id, title, path))
            return {
                "id": "doc1",
                "group_id": group_id,
                "document_title": title,
                "file_path": path
            }

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
        assert 'group-documents/test-group-id/' in captured_args[0][2]  # storage path

        # Check response includes file_size and mime_type
        json_data = response.get_json()
        assert json_data['file_size'] == 1024
        assert json_data['mime_type'] == 'application/pdf'


class TestFileServing:
    """Test file serving via Supabase Storage redirects."""

    def test_serve_uploaded_file_redirects_to_storage(self, client, monkeypatch):
        """Test that file requests redirect to Supabase Storage public URL."""
        def mock_get_url(storage_path):
            return f"https://test.supabase.co/storage/v1/object/public/{storage_path}"

        monkeypatch.setattr('api.index.get_public_url', mock_get_url)

        response = client.get('/uploads/group-documents/group1/test_file.pdf')

        # Should redirect to Supabase Storage URL
        assert response.status_code == 302
        assert 'https://test.supabase.co/storage/v1/object/public/' in response.location
        assert 'group-documents/group1/test_file.pdf' in response.location

    def test_serve_file_with_missing_storage_path(self, client, monkeypatch):
        """Test serving a file when storage returns None."""
        def mock_get_url_none(storage_path):
            return None

        monkeypatch.setattr('api.index.get_public_url', mock_get_url_none)

        response = client.get('/uploads/group-documents/nonexistent/file.pdf')
        # Should return 404 when public URL cannot be generated
        assert response.status_code == 404

    def test_serve_file_path_traversal_blocked(self, client):
        """Test that path traversal attempts are blocked."""
        # Try to access file with path traversal
        response = client.get('/uploads/../../../etc/passwd')

        # Should be blocked (400, 404, or redirect to a safe error)
        assert response.status_code in [400, 404, 302]


class TestGroupDeletion:
    """Test group deletion functionality with Supabase Storage."""

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

    def test_delete_group_removes_storage_files(self, admin_client, mock_supabase, monkeypatch):
        """Test that deleting a group removes files from Supabase Storage."""
        deleted_files = []

        def mock_delete_storage(storage_path):
            deleted_files.append(storage_path)
            return True

        monkeypatch.setattr('api.index.delete_file_from_storage', mock_delete_storage)

        # Mock group with documents in Supabase Storage
        mock_group = {
            'id': 'test-group-id',
            'group_name': 'Test Group',
            'documents': [
                {'file_path': 'group-documents/group1/file1.pdf', 'document_title': 'File 1'},
                {'file_path': 'group-documents/group1/file2.pdf', 'document_title': 'File 2'}
            ]
        }

        monkeypatch.setattr('api.index.get_group_details', lambda x: mock_group)
        monkeypatch.setattr('api.index.delete_group', lambda x: True)

        response = admin_client.delete('/api/groups/test-group-id')

        assert response.status_code == 200
        # Verify files were deleted from storage
        assert len(deleted_files) == 2
        assert 'group-documents/group1/file1.pdf' in deleted_files
        assert 'group-documents/group1/file2.pdf' in deleted_files

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
    """Integration tests for group portal features with Supabase Storage."""

    def test_upload_and_retrieve_document(self, client, mock_supabase, monkeypatch):
        """Test full workflow: upload document and retrieve via Supabase Storage URL."""
        # Mock storage helper
        def mock_upload(file, group_id, document_title):
            return (f"group-documents/{group_id}/test_file.pdf", 1024, "application/pdf")

        def mock_get_url(storage_path):
            return f"https://test.supabase.co/storage/v1/object/public/{storage_path}"

        monkeypatch.setattr('api.index.upload_file_to_storage', mock_upload)
        monkeypatch.setattr('api.index.get_public_url', mock_get_url)

        # Capture the storage path that gets saved
        saved_storage_path = None

        def mock_add_doc(group_id, title, path):
            nonlocal saved_storage_path
            saved_storage_path = path
            return {
                "id": "doc1",
                "group_id": group_id,
                "document_title": title,
                "file_path": path
            }

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
        assert saved_storage_path is not None
        assert 'group-documents/integration-group/' in saved_storage_path

        # Retrieve the file (should redirect to Supabase Storage)
        retrieve_response = client.get(f'/uploads/{saved_storage_path}')

        assert retrieve_response.status_code == 302  # Redirect
        assert 'https://test.supabase.co/storage/v1/object/public/' in retrieve_response.location
        assert saved_storage_path in retrieve_response.location


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
