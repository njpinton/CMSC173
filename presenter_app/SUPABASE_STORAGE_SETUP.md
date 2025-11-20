# Supabase Storage Setup Guide

This guide provides step-by-step instructions for setting up Supabase Storage for the CMSC 173 Presenter App.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [SQL Editor Setup](#sql-editor-setup)
3. [Verify Storage Setup](#verify-storage-setup)
4. [Test File Upload](#test-file-upload)
5. [Environment Configuration](#environment-configuration)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:
- ✅ A Supabase project created
- ✅ Access to your Supabase project dashboard
- ✅ Your Supabase URL and anon key (from Project Settings > API)

---

## SQL Editor Setup

### Step 1: Access the SQL Editor

1. Log in to your Supabase project dashboard at [https://supabase.com](https://supabase.com)
2. Select your project
3. Click on **SQL Editor** in the left sidebar
4. Click **New query** button

### Step 2: Run Storage Bucket Creation Script

Copy and paste the following SQL commands into the SQL Editor:

```sql
-- ============================================
-- SUPABASE STORAGE SETUP
-- ============================================
-- This script creates the storage bucket and sets up access policies
-- for the CMSC 173 Presenter App

-- Step 1: Create the storage bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('group-documents', 'group-documents', true)
ON CONFLICT (id) DO NOTHING;

-- Step 2: Set up storage policies for public read access
CREATE POLICY "Public Access for group documents"
ON storage.objects FOR SELECT
USING (bucket_id = 'group-documents');

-- Step 3: Allow authenticated uploads (you can adjust this)
CREATE POLICY "Authenticated users can upload group documents"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'group-documents');

-- Step 4: Allow authenticated updates
CREATE POLICY "Authenticated users can update their documents"
ON storage.objects FOR UPDATE
USING (bucket_id = 'group-documents');

-- Step 5: Allow authenticated deletions
CREATE POLICY "Authenticated users can delete group documents"
ON storage.objects FOR DELETE
USING (bucket_id = 'group-documents');
```

3. Click **Run** or press `Cmd/Ctrl + Enter`
4. You should see success messages confirming:
   - Bucket created: `group-documents`
   - 4 policies created

### Step 3: Update Database Schema (If Not Done Already)

If you haven't already set up your database tables, run the main schema:

1. Open a new SQL Editor query
2. Copy the entire contents of `supabase_schema.sql`
3. Paste into the SQL Editor
4. Click **Run**

This creates all necessary tables:
- `module_views`
- `groups`
- `group_members`
- `group_documents`

---

## Verify Storage Setup

### Option 1: Using SQL Queries

Run these verification queries in the SQL Editor:

```sql
-- Check if bucket exists
SELECT * FROM storage.buckets WHERE id = 'group-documents';

-- Check storage policies
SELECT * FROM pg_policies
WHERE schemaname = 'storage'
AND tablename = 'objects'
AND policyname LIKE '%group documents%';

-- Count policies (should show 4)
SELECT COUNT(*) as policy_count
FROM pg_policies
WHERE schemaname = 'storage'
AND tablename = 'objects'
AND policyname LIKE '%group documents%';
```

Expected results:
- ✅ Bucket `group-documents` exists with `public = true`
- ✅ 4 policies created (SELECT, INSERT, UPDATE, DELETE)

### Option 2: Using Supabase Dashboard

1. Go to **Storage** in the left sidebar
2. You should see the `group-documents` bucket listed
3. Click on the bucket name
4. You should see an empty bucket ready for file uploads

---

## Test File Upload

### Test Using the Supabase Dashboard

1. Go to **Storage** > `group-documents` bucket
2. Click **Upload file**
3. Create a test folder: `test-group/`
4. Upload a sample PDF file
5. Verify you can see the file
6. Click on the file to get the public URL
7. Open the URL in a browser - it should display/download

### Test Using Your Application

Once your app is running:

1. Navigate to the Group Portal
2. Create a test group
3. Upload a document
4. Verify the file appears in Supabase Storage:
   - Go to Storage > `group-documents`
   - You should see a folder named after your group ID
   - Inside, you should see the uploaded file with a timestamp prefix

---

## Environment Configuration

### Local Development (.env file)

Ensure your `.env` file has these variables:

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

### Vercel Deployment

If deploying to Vercel:

1. Go to your Vercel project settings
2. Navigate to **Environment Variables**
3. Add these variables:
   - `SUPABASE_URL` = `https://your-project-id.supabase.co`
   - `SUPABASE_ANON_KEY` = `your-anon-key-here`
   - `FLASK_SECRET_KEY` = `your-secret-key`
   - `ADMIN_USERNAME` = `admin`
   - `ADMIN_PASSWORD_HASH` = `your-hashed-password`

---

## Storage Structure

Your files will be organized in Supabase Storage as follows:

```
group-documents/
├── {group-id-1}/
│   ├── 1234567890_proposal.pdf
│   ├── 1234567891_final-report.pdf
│   └── 1234567892_presentation.pptx
├── {group-id-2}/
│   ├── 1234567893_document.pdf
│   └── 1234567894_slides.pdf
└── {group-id-3}/
    └── 1234567895_paper.pdf
```

**File naming format**: `{timestamp}_{original-filename}`

**Storage path format**: `group-documents/{group-id}/{timestamp}_{filename}`

**Public URL format**:
```
https://{your-project-id}.supabase.co/storage/v1/object/public/group-documents/{group-id}/{timestamp}_{filename}
```

---

## Troubleshooting

### Error: "Bucket already exists"

**Problem**: Running the setup script multiple times

**Solution**: This is safe to ignore. The `ON CONFLICT DO NOTHING` clause prevents errors.

### Error: "Policy already exists"

**Problem**: Policies were created in a previous run

**Solution**: Either:
1. Drop existing policies first:
   ```sql
   DROP POLICY IF EXISTS "Public Access for group documents" ON storage.objects;
   DROP POLICY IF EXISTS "Authenticated users can upload group documents" ON storage.objects;
   DROP POLICY IF EXISTS "Authenticated users can update their documents" ON storage.objects;
   DROP POLICY IF EXISTS "Authenticated users can delete group documents" ON storage.objects;
   ```
   Then re-run the creation script.

2. Or simply ignore the error if policies are already set up correctly.

### Error: "Failed to upload file to storage"

**Possible causes**:

1. **Invalid Supabase credentials**
   - Check `SUPABASE_URL` and `SUPABASE_ANON_KEY` in your `.env`
   - Verify they match your Supabase project settings

2. **Bucket doesn't exist**
   - Verify bucket exists: Go to Storage in Supabase dashboard
   - Re-run the bucket creation script

3. **File size too large**
   - Default Supabase limit: 50 MB for free tier
   - Check your tier limits in Supabase dashboard

4. **Network/connectivity issues**
   - Check your internet connection
   - Verify Supabase service status

### Files Not Appearing in Storage

**Problem**: Upload succeeds but files don't appear

**Solution**:
1. Check the database `group_documents` table:
   ```sql
   SELECT * FROM group_documents ORDER BY created_at DESC LIMIT 10;
   ```
2. Verify `file_path` format: `group-documents/{group-id}/{filename}`
3. Navigate to Storage > `group-documents` and look for the group folder

### Public URLs Not Working

**Problem**: Public URLs return 404 or access denied

**Solution**:
1. Verify bucket is public:
   ```sql
   SELECT id, name, public FROM storage.buckets WHERE id = 'group-documents';
   ```
   Should show `public = true`

2. Check SELECT policy exists:
   ```sql
   SELECT * FROM pg_policies
   WHERE tablename = 'objects'
   AND policyname = 'Public Access for group documents';
   ```

3. Ensure file path is correct in database

### Permission Errors

**Problem**: "Permission denied" when uploading

**Solution**:
1. Verify INSERT policy exists for the bucket
2. Check your anon key has proper permissions
3. If using RLS, ensure policies allow anonymous uploads

---

## Security Considerations

### Recommended Settings for Production

1. **Enable RLS on storage.objects** (done by default)
2. **Limit file types**: Implement file type validation in your app
3. **Limit file sizes**: Set `MAX_FILE_SIZE` in your application
4. **Consider authenticated-only uploads**: Modify the INSERT policy:
   ```sql
   DROP POLICY "Authenticated users can upload group documents" ON storage.objects;

   CREATE POLICY "Only authenticated users can upload"
   ON storage.objects FOR INSERT
   TO authenticated
   WITH CHECK (bucket_id = 'group-documents');
   ```

5. **Enable virus scanning** (available in Supabase Pro tier)

### File Path Security

The application implements:
- ✅ Secure filename generation with timestamps
- ✅ Path traversal prevention
- ✅ Input validation for group IDs
- ✅ MIME type validation

---

## Additional Resources

- [Supabase Storage Documentation](https://supabase.com/docs/guides/storage)
- [Row Level Security Policies](https://supabase.com/docs/guides/auth/row-level-security)
- [Storage API Reference](https://supabase.com/docs/reference/javascript/storage)

---

## Quick Reference: Common SQL Commands

### List all buckets
```sql
SELECT * FROM storage.buckets;
```

### List all files in a bucket
```sql
SELECT * FROM storage.objects
WHERE bucket_id = 'group-documents'
ORDER BY created_at DESC;
```

### Count files per group
```sql
SELECT
    split_part(name, '/', 1) as group_id,
    COUNT(*) as file_count
FROM storage.objects
WHERE bucket_id = 'group-documents'
GROUP BY split_part(name, '/', 1);
```

### Get total storage used
```sql
SELECT
    bucket_id,
    COUNT(*) as file_count,
    pg_size_pretty(SUM(COALESCE((metadata->>'size')::bigint, 0))) as total_size
FROM storage.objects
WHERE bucket_id = 'group-documents'
GROUP BY bucket_id;
```

### Delete all files for a specific group
```sql
DELETE FROM storage.objects
WHERE bucket_id = 'group-documents'
AND name LIKE 'your-group-id/%';
```

---

**Setup Complete!** 🎉

Your Supabase Storage is now configured and ready to handle file uploads for the presenter app.

For issues or questions, check the troubleshooting section or consult the Supabase documentation.

**Last Updated**: November 2024
**Version**: 1.0.0
