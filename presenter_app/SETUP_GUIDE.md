# CMSC 173 Presenter App - Setup Guide

This guide will walk you through setting up the Supabase database and configuring the admin dashboard.

## Table of Contents
1. [Supabase Database Setup](#supabase-database-setup)
2. [Environment Configuration](#environment-configuration)
3. [Admin Dashboard Access](#admin-dashboard-access)
4. [Testing](#testing)

---

## Supabase Database Setup

### Step 1: Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign in/sign up
2. Click "New Project"
3. Choose your organization
4. Fill in:
   - **Project Name**: `cmsc173-presenter-app` (or your choice)
   - **Database Password**: Choose a strong password (save it!)
   - **Region**: Choose closest to your location
5. Click "Create new project" and wait for it to initialize (~2 minutes)

### Step 2: Run the Database Schema

1. In your Supabase project, go to **SQL Editor** (left sidebar)
2. Open the file `supabase_schema.sql` in this directory
3. Copy the **entire contents** of the file
4. Paste it into the SQL Editor
5. Click **Run** (or press Cmd/Ctrl + Enter)

You should see success messages for all table creations.

### Step 3: Verify Tables Were Created

Run this query in the SQL Editor to verify:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('module_views', 'groups', 'group_members', 'group_documents');
```

You should see all 4 tables listed.

### Step 4: Get Your API Keys

1. Go to **Project Settings** (gear icon in sidebar)
2. Click **API** in the settings menu
3. Copy these values:
   - **Project URL** (looks like `https://xxxxx.supabase.co`)
   - **anon public** key (long string starting with `eyJ...`)

---

## Environment Configuration

### Local Development (.env file)

Create a `.env` file in the `presenter_app` directory:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here

# Flask Configuration
FLASK_SECRET_KEY=your-random-secret-key-here

# Admin Credentials
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=your-hashed-password-here
```

#### Generating the Admin Password Hash

Run this Python command to generate a password hash:

```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password_here'))"
```

Replace `your_password_here` with your desired admin password.

Copy the output and paste it as the `ADMIN_PASSWORD_HASH` value.

### Vercel Deployment

If deploying to Vercel:

1. Go to your Vercel project settings
2. Click **Environment Variables**
3. Add these variables:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `FLASK_SECRET_KEY`
   - `ADMIN_USERNAME`
   - `ADMIN_PASSWORD_HASH`
   - `ALLOWED_ORIGINS` (e.g., `https://your-app.vercel.app`)

---

## Admin Dashboard Access

### Logging In

1. Navigate to `/admin_login`
2. Enter your admin username and password
3. Click "Login"
4. You'll be redirected to the Group Portal

### Accessing the Dashboard

Once logged in as admin, you'll see a **📊 Dashboard** button in the top right of the Group Portal.

The admin dashboard shows:
- **Statistics**: Total groups, students, submissions, and averages
- **All Groups**: Complete list with member and document details
- **Group Management**: Delete groups and view all submissions

### Features Available to Admins

✅ **View all groups and their members**
✅ **View all document submissions**
✅ **Delete groups** (removes database records and physical files)
✅ **Real-time statistics**
✅ **Download submitted documents**

---

## Testing

### Running Tests

```bash
cd presenter_app

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest test_*.py -v

# Run specific test suites
pytest test_admin_visibility.py -v  # Admin visibility tests
pytest test_group_portal.py -v       # Group portal tests
pytest test_security_features.py -v  # Security tests
```

### Manual Testing

1. **Create a test group**:
   - Go to `/group_portal`
   - Fill in group name and add members
   - Click "Create Group"

2. **Upload a document**:
   - Click "View Details" on a group
   - Click "Upload Document"
   - Select a file and submit

3. **Access admin dashboard**:
   - Login at `/admin_login`
   - Navigate to `/admin_dashboard`
   - Verify you can see all groups and documents

4. **Test admin features**:
   - Try deleting a group
   - Download a submitted document
   - Check statistics update correctly

---

## Database Schema Overview

### Tables

1. **`module_views`**: Tracks module view counts
   - `module_number` (INT, PRIMARY KEY)
   - `view_count` (INT)
   - `created_at`, `updated_at` (TIMESTAMP)

2. **`groups`**: Student groups
   - `id` (UUID, PRIMARY KEY)
   - `group_name` (VARCHAR 100)
   - `project_title` (VARCHAR 255)
   - `created_at`, `updated_at` (TIMESTAMP)

3. **`group_members`**: Group membership
   - `id` (UUID, PRIMARY KEY)
   - `group_id` (UUID, FOREIGN KEY)
   - `member_name` (VARCHAR 100)
   - `created_at` (TIMESTAMP)

4. **`group_documents`**: Uploaded documents
   - `id` (UUID, PRIMARY KEY)
   - `group_id` (UUID, FOREIGN KEY)
   - `document_title` (VARCHAR 255)
   - `file_path` (TEXT)
   - `created_at` (TIMESTAMP)

### Functions

- **`increment_module_view(module_id INT)`**: Increments view count for a module

### Views

- **`group_summary`**: Aggregates group data with member/document counts

---

## Troubleshooting

### "Supabase not configured" Error

**Problem**: Missing or incorrect Supabase environment variables

**Solution**:
1. Check your `.env` file has `SUPABASE_URL` and `SUPABASE_ANON_KEY`
2. Verify the values are correct (no extra spaces)
3. Restart your Flask app

### Admin Login Failed

**Problem**: Invalid credentials or incorrect password hash

**Solution**:
1. Verify `ADMIN_USERNAME` matches what you're entering
2. Regenerate `ADMIN_PASSWORD_HASH`:
   ```bash
   python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password'))"
   ```
3. Update the environment variable
4. Restart the app

### Files Not Uploading

**Problem**: Upload directory doesn't exist or has wrong permissions

**Solution**:
1. Check that `uploads/` directory exists in project root
2. Ensure it has write permissions:
   ```bash
   mkdir -p uploads
   chmod 755 uploads
   ```

### Database Connection Issues

**Problem**: Can't connect to Supabase

**Solution**:
1. Check your internet connection
2. Verify Supabase project is active (not paused)
3. Check API keys are correct
4. Look at Supabase project logs for errors

---

## Security Notes

⚠️ **Important Security Considerations**:

1. **Never commit** `.env` files to version control
2. **Use strong passwords** for admin accounts
3. **Rotate API keys** regularly
4. **Enable Row Level Security (RLS)** on all Supabase tables
5. **Use HTTPS** in production
6. **Set proper CORS origins** for production

---

## Support

For issues or questions:
1. Check the logs for error messages
2. Review the test suite for examples
3. Check Supabase project logs
4. Verify all environment variables are set correctly

---

**Last Updated**: November 2024
**Version**: 1.0.0
