"""
Test suite for admin dashboard functionality
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


@pytest.fixture
def admin_client(client):
    """Create a client with admin session."""
    client.post(
        '/admin_login',
        data={'username': 'testadmin', 'password': 'testpassword123'}
    )
    return client


class TestAdminDashboardAccess:
    """Test admin dashboard access control."""

    def test_dashboard_requires_authentication(self, client):
        """Test that dashboard requires admin authentication."""
        response = client.get('/admin_dashboard')

        # Should return 403 Forbidden without admin auth
        assert response.status_code == 403

    def test_dashboard_accessible_to_admin(self, admin_client):
        """Test that admin can access dashboard."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        assert b'Admin Dashboard' in response.data

    def test_dashboard_shows_statistics_section(self, admin_client):
        """Test that dashboard shows statistics section."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        assert b'Total Groups' in response.data
        assert b'Total Students' in response.data
        assert b'Total Submissions' in response.data

    def test_dashboard_has_navigation_buttons(self, admin_client):
        """Test that dashboard has proper navigation."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        assert b'Group Portal' in response.data
        assert b'Logout' in response.data
        assert b'Modules' in response.data

    def test_dashboard_has_refresh_functionality(self, admin_client):
        """Test that dashboard has refresh button."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        assert b'refresh' in response.data.lower() or '↻'.encode('utf-8') in response.data


class TestGroupPortalDashboardLink:
    """Test dashboard link in group portal for admin users."""

    def test_group_portal_shows_dashboard_link_for_admin(self, admin_client):
        """Test that admin sees dashboard link in group portal."""
        response = admin_client.get('/group_portal')

        assert response.status_code == 200
        assert b'Dashboard' in response.data
        assert b'/admin_dashboard' in response.data

    def test_group_portal_no_dashboard_link_for_non_admin(self, client):
        """Test that non-admin doesn't see dashboard link."""
        response = client.get('/group_portal')

        assert response.status_code == 200
        # Should not contain dashboard link
        assert b'/admin_dashboard' not in response.data


class TestAdminDashboardContent:
    """Test admin dashboard content and functionality."""

    def test_dashboard_loads_groups_section(self, admin_client):
        """Test that dashboard has groups section."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        assert b'All Groups' in response.data or b'Submissions' in response.data
        assert b'groupsContainer' in response.data

    def test_dashboard_has_javascript_functions(self, admin_client):
        """Test that dashboard includes necessary JavaScript functions."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        assert b'loadDashboardData' in response.data
        assert b'loadGroupDetails' in response.data
        assert b'deleteGroup' in response.data

    def test_dashboard_fetches_groups_on_load(self, admin_client):
        """Test that dashboard includes code to fetch groups on load."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        assert b'/api/groups' in response.data
        assert b'DOMContentLoaded' in response.data


class TestAdminDashboardLogout:
    """Test logout functionality from admin dashboard."""

    def test_logout_from_dashboard_clears_session(self, admin_client):
        """Test that logging out from dashboard clears admin session."""
        # First verify we can access dashboard
        response1 = admin_client.get('/admin_dashboard')
        assert response1.status_code == 200

        # Logout
        admin_client.get('/admin_logout')

        # Try to access dashboard again
        response2 = admin_client.get('/admin_dashboard')
        assert response2.status_code == 403  # Should be forbidden now


class TestAdminDashboardIntegration:
    """Integration tests for admin dashboard."""

    def test_full_admin_workflow(self, client, monkeypatch):
        """Test complete admin workflow: login -> dashboard -> logout."""
        from unittest.mock import MagicMock

        # Mock get_groups to return sample data
        mock_groups = [
            {
                'id': 'group1',
                'group_name': 'Test Group',
                'project_title': 'Test Project',
                'created_at': '2024-01-01T00:00:00Z'
            }
        ]
        monkeypatch.setattr('api.index.get_groups', lambda: mock_groups)

        # Login
        login_response = client.post(
            '/admin_login',
            data={'username': 'testadmin', 'password': 'testpassword123'}
        )
        assert login_response.status_code == 302

        # Access dashboard
        dashboard_response = client.get('/admin_dashboard')
        assert dashboard_response.status_code == 200
        assert b'Admin Dashboard' in dashboard_response.data

        # Logout
        logout_response = client.get('/admin_logout')
        assert logout_response.status_code == 302

        # Verify can't access dashboard anymore
        dashboard_response2 = client.get('/admin_dashboard')
        assert dashboard_response2.status_code == 403

    def test_dashboard_shows_empty_state_for_no_groups(self, admin_client, monkeypatch):
        """Test that dashboard shows appropriate message when no groups exist."""
        # Mock get_groups to return empty list
        monkeypatch.setattr('api.index.get_groups', lambda: [])

        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        # Dashboard should still load, JavaScript will handle empty state
        assert b'loadDashboardData' in response.data


class TestAdminDashboardSecurity:
    """Test security aspects of admin dashboard."""

    def test_dashboard_escapes_html_in_javascript(self, admin_client):
        """Test that dashboard uses HTML escaping in JavaScript."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        # Check for escapeHtml function
        assert b'escapeHtml' in response.data

    def test_dashboard_uses_safe_delete_confirmation(self, admin_client):
        """Test that delete operations require confirmation."""
        response = admin_client.get('/admin_dashboard')

        assert response.status_code == 200
        # Check for confirm dialog in delete function
        assert b'confirm' in response.data
        assert b'deleteGroup' in response.data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
