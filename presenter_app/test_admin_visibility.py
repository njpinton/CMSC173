"""
Test admin visibility in group portal
"""

import pytest
import os
from werkzeug.security import generate_password_hash

# Set required environment variables before importing the app
os.environ['FLASK_SECRET_KEY'] = 'test_secret_key_for_testing_only'
os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
os.environ['SUPABASE_ANON_KEY'] = 'test_anon_key'
os.environ['VERCEL_ENV'] = 'development'
os.environ['ADMIN_USERNAME'] = 'testadmin'
os.environ['ADMIN_PASSWORD_HASH'] = generate_password_hash('testpassword123')

from api.index import app


@pytest.fixture
def client(monkeypatch):
    """Create Flask test client."""
    app.config['TESTING'] = True
    from api.index import limiter
    monkeypatch.setattr(limiter, 'enabled', False)

    from unittest.mock import MagicMock
    mock_client = MagicMock()
    monkeypatch.setattr('api.index.get_supabase_client', lambda: mock_client)

    with app.test_client() as test_client:
        yield test_client


def test_group_portal_shows_admin_login_when_not_logged_in(client):
    """Test that group portal shows 'Admin Login' button when not logged in."""
    response = client.get('/group_portal')

    assert response.status_code == 200
    assert b'Admin Login' in response.data
    assert b'Logout Admin' not in response.data


def test_group_portal_shows_logout_when_admin_logged_in(client):
    """Test that group portal shows 'Logout Admin' button when admin is logged in."""
    # Login as admin
    client.post(
        '/admin_login',
        data={'username': 'testadmin', 'password': 'testpassword123'}
    )

    # Access group portal
    response = client.get('/group_portal')

    assert response.status_code == 200
    assert b'Logout Admin' in response.data
    assert b'Admin Login' not in response.data


def test_admin_can_see_delete_buttons(client):
    """Test that admin users can see delete buttons in JavaScript."""
    # Login as admin
    client.post(
        '/admin_login',
        data={'username': 'testadmin', 'password': 'testpassword123'}
    )

    # Access group portal
    response = client.get('/group_portal')

    assert response.status_code == 200
    # Check that IS_ADMIN JavaScript variable is set to true
    assert b'const IS_ADMIN = true' in response.data


def test_non_admin_cannot_see_delete_buttons(client):
    """Test that non-admin users don't see delete buttons in JavaScript."""
    # Access group portal without logging in
    response = client.get('/group_portal')

    assert response.status_code == 200
    # Check that IS_ADMIN JavaScript variable is set to false
    assert b'const IS_ADMIN = false' in response.data


def test_admin_status_persists_across_requests(client):
    """Test that admin status persists across multiple requests."""
    # Login as admin
    login_response = client.post(
        '/admin_login',
        data={'username': 'testadmin', 'password': 'testpassword123'}
    )
    assert login_response.status_code == 302

    # First request to group portal
    response1 = client.get('/group_portal')
    assert b'Logout Admin' in response1.data

    # Second request to group portal
    response2 = client.get('/group_portal')
    assert b'Logout Admin' in response2.data

    # Admin status should still be there
    assert b'const IS_ADMIN = true' in response2.data


def test_admin_status_cleared_after_logout(client):
    """Test that admin status is cleared after logout."""
    # Login as admin
    client.post(
        '/admin_login',
        data={'username': 'testadmin', 'password': 'testpassword123'}
    )

    # Verify admin status
    response1 = client.get('/group_portal')
    assert b'Logout Admin' in response1.data

    # Logout
    client.get('/admin_logout')

    # Check that admin status is gone
    response2 = client.get('/group_portal')
    assert b'Admin Login' in response2.data
    assert b'Logout Admin' not in response2.data
    assert b'const IS_ADMIN = false' in response2.data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
