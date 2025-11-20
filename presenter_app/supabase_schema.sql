-- ============================================
-- CMSC 173 Presenter App - Supabase Database Schema
-- ============================================

-- This file contains the complete database schema for the presenter app
-- Run these SQL commands in your Supabase SQL Editor

-- ============================================
-- 0. STORAGE SETUP (Run this first!)
-- ============================================
-- Create storage bucket for group documents

-- Create the storage bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('group-documents', 'group-documents', true)
ON CONFLICT (id) DO NOTHING;

-- Set up storage policies for public access
CREATE POLICY "Public Access for group documents"
ON storage.objects FOR SELECT
USING (bucket_id = 'group-documents');

CREATE POLICY "Authenticated users can upload group documents"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'group-documents');

CREATE POLICY "Authenticated users can update their documents"
ON storage.objects FOR UPDATE
USING (bucket_id = 'group-documents');

CREATE POLICY "Authenticated users can delete group documents"
ON storage.objects FOR DELETE
USING (bucket_id = 'group-documents');

-- ============================================
-- 1. MODULE VIEWS TABLE
-- ============================================
-- Tracks how many times each module has been viewed

CREATE TABLE IF NOT EXISTS module_views (
    module_number INT PRIMARY KEY,
    view_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create function to increment module view count
CREATE OR REPLACE FUNCTION increment_module_view(module_id INT)
RETURNS VOID AS $$
BEGIN
    INSERT INTO module_views (module_number, view_count)
    VALUES (module_id, 1)
    ON CONFLICT (module_number)
    DO UPDATE SET
        view_count = module_views.view_count + 1,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 2. GROUPS TABLE
-- ============================================
-- Stores information about student groups

CREATE TABLE IF NOT EXISTS groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_name VARCHAR(100) NOT NULL,
    project_title VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_groups_created_at ON groups(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_groups_name ON groups(group_name);

-- ============================================
-- 3. GROUP MEMBERS TABLE
-- ============================================
-- Stores members of each group

CREATE TABLE IF NOT EXISTS group_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    member_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_group_members_group_id ON group_members(group_id);
CREATE INDEX IF NOT EXISTS idx_group_members_name ON group_members(member_name);

-- ============================================
-- 4. GROUP DOCUMENTS TABLE
-- ============================================
-- Stores documents uploaded by groups
-- file_path now stores Supabase Storage path (e.g., 'group-documents/groupid/filename.pdf')
-- Full public URL is constructed as: {SUPABASE_URL}/storage/v1/object/public/{file_path}

CREATE TABLE IF NOT EXISTS group_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    document_title VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,  -- Supabase Storage path
    file_size BIGINT,
    mime_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_group_documents_group_id ON group_documents(group_id);
CREATE INDEX IF NOT EXISTS idx_group_documents_created_at ON group_documents(created_at DESC);

-- ============================================
-- 5. ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================
-- Enable RLS on all tables for security

-- Module Views - Public read access
ALTER TABLE module_views ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access to module_views"
ON module_views FOR SELECT
TO public
USING (true);

CREATE POLICY "Allow public insert/update to module_views"
ON module_views FOR ALL
TO public
USING (true)
WITH CHECK (true);

-- Groups - Public read/write access (adjust based on your needs)
ALTER TABLE groups ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access to groups"
ON groups FOR SELECT
TO public
USING (true);

CREATE POLICY "Allow public insert to groups"
ON groups FOR INSERT
TO public
WITH CHECK (true);

CREATE POLICY "Allow public update to groups"
ON groups FOR UPDATE
TO public
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow public delete to groups"
ON groups FOR DELETE
TO public
USING (true);

-- Group Members - Public read/write access
ALTER TABLE group_members ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public access to group_members"
ON group_members FOR ALL
TO public
USING (true)
WITH CHECK (true);

-- Group Documents - Public read/write access
ALTER TABLE group_documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public access to group_documents"
ON group_documents FOR ALL
TO public
USING (true)
WITH CHECK (true);

-- ============================================
-- 6. HELPFUL VIEWS
-- ============================================
-- View to get group summary with member and document counts

CREATE OR REPLACE VIEW group_summary AS
SELECT
    g.id,
    g.group_name,
    g.project_title,
    g.created_at,
    g.updated_at,
    COUNT(DISTINCT gm.id) as member_count,
    COUNT(DISTINCT gd.id) as document_count
FROM groups g
LEFT JOIN group_members gm ON g.id = gm.group_id
LEFT JOIN group_documents gd ON g.id = gd.group_id
GROUP BY g.id, g.group_name, g.project_title, g.created_at, g.updated_at;

-- ============================================
-- 7. UPDATED_AT TRIGGER
-- ============================================
-- Automatically update updated_at timestamp

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_groups_updated_at
    BEFORE UPDATE ON groups
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 8. SAMPLE DATA (Optional - for testing)
-- ============================================
-- Uncomment to insert sample data

/*
-- Insert sample module views
INSERT INTO module_views (module_number, view_count) VALUES
(0, 150),
(1, 142),
(2, 138),
(3, 125),
(4, 118),
(5, 110),
(6, 105),
(7, 98),
(8, 92),
(9, 85),
(10, 78),
(11, 72),
(12, 65),
(13, 58)
ON CONFLICT (module_number) DO NOTHING;

-- Insert sample group
INSERT INTO groups (id, group_name, project_title) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Team Alpha', 'Image Classification with CNNs')
ON CONFLICT (id) DO NOTHING;

-- Insert sample members
INSERT INTO group_members (group_id, member_name) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Alice Johnson'),
('550e8400-e29b-41d4-a716-446655440001', 'Bob Smith'),
('550e8400-e29b-41d4-a716-446655440001', 'Charlie Davis')
ON CONFLICT (id) DO NOTHING;

-- Insert sample document
INSERT INTO group_documents (group_id, document_title, file_path) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Project Proposal', '/uploads/1234567890_proposal.pdf')
ON CONFLICT (id) DO NOTHING;
*/

-- ============================================
-- VERIFICATION QUERIES
-- ============================================
-- Run these to verify your setup

-- Check all tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('module_views', 'groups', 'group_members', 'group_documents');

-- Check RLS is enabled
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN ('module_views', 'groups', 'group_members', 'group_documents');

-- View group summary
SELECT * FROM group_summary;

-- ============================================
-- SETUP COMPLETE!
-- ============================================
-- Your database is now ready to use with the presenter app.
--
-- Next steps:
-- 1. Copy your Supabase URL and anon key from:
--    Project Settings > API
--
-- 2. Set environment variables:
--    SUPABASE_URL=your_url_here
--    SUPABASE_ANON_KEY=your_key_here
--
-- 3. For admin features, also set:
--    ADMIN_USERNAME=your_username
--    ADMIN_PASSWORD_HASH=your_hashed_password
--    FLASK_SECRET_KEY=your_secret_key
