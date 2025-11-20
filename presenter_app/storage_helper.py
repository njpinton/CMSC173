"""
Supabase Storage helper functions for file uploads
Handles uploading files to Supabase Storage bucket
"""

import os
import logging
from typing import Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Storage bucket name
BUCKET_NAME = "group-documents"


def upload_file_to_storage(
    file: FileStorage,
    group_id: str,
    document_title: str
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Upload a file to Supabase Storage.

    Args:
        file: The file object from Flask request
        group_id: The group ID for organizing files
        document_title: The title of the document

    Returns:
        Tuple of (storage_path, file_size, mime_type) or (None, None, None) on error
        storage_path format: 'group-documents/{group_id}/{timestamp}_{filename}'
    """
    supabase = get_supabase_client()

    if not supabase:
        logger.error("Supabase client not initialized for file upload")
        return None, None, None

    try:
        # Get file info
        import time
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"

        # Construct storage path: group-id/timestamp_filename
        storage_path = f"{group_id}/{unique_filename}"

        # Read file content
        file_content = file.read()
        file_size = len(file_content)
        mime_type = file.content_type or 'application/octet-stream'

        # Reset file pointer in case it's needed again
        file.seek(0)

        # Upload to Supabase Storage
        logger.info(f"Uploading file to storage: {BUCKET_NAME}/{storage_path}")

        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=file_content,
            file_options={
                "content-type": mime_type,
                "cache-control": "3600",
                "upsert": "false"
            }
        )

        logger.info(f"File uploaded successfully: {storage_path}")

        # Return the storage path (not full URL, we'll construct that when needed)
        return f"{BUCKET_NAME}/{storage_path}", file_size, mime_type

    except Exception as e:
        logger.error(f"Error uploading file to storage: {e}", exc_info=True)
        return None, None, None


def delete_file_from_storage(storage_path: str) -> bool:
    """
    Delete a file from Supabase Storage.

    Args:
        storage_path: The storage path (e.g., 'group-documents/groupid/file.pdf')

    Returns:
        True if successful, False otherwise
    """
    supabase = get_supabase_client()

    if not supabase:
        logger.error("Supabase client not initialized for file deletion")
        return False

    try:
        # Extract bucket and path
        if '/' not in storage_path:
            logger.error(f"Invalid storage path format: {storage_path}")
            return False

        # storage_path format: 'group-documents/groupid/filename'
        # We need to extract bucket and file path
        parts = storage_path.split('/', 1)
        if len(parts) != 2:
            logger.error(f"Invalid storage path format: {storage_path}")
            return False

        bucket = parts[0]
        file_path = parts[1]

        logger.info(f"Deleting file from storage: {bucket}/{file_path}")

        response = supabase.storage.from_(bucket).remove([file_path])

        logger.info(f"File deleted successfully: {storage_path}")
        return True

    except Exception as e:
        logger.error(f"Error deleting file from storage: {e}", exc_info=True)
        return False


def get_public_url(storage_path: str) -> Optional[str]:
    """
    Get the public URL for a file in Supabase Storage.

    Args:
        storage_path: The storage path (e.g., 'group-documents/groupid/file.pdf')

    Returns:
        Public URL or None if error
    """
    supabase = get_supabase_client()

    if not supabase:
        logger.error("Supabase client not initialized")
        return None

    try:
        # storage_path format: 'group-documents/groupid/filename'
        parts = storage_path.split('/', 1)
        if len(parts) != 2:
            logger.error(f"Invalid storage path format: {storage_path}")
            return None

        bucket = parts[0]
        file_path = parts[1]

        # Get public URL
        public_url = supabase.storage.from_(bucket).get_public_url(file_path)

        return public_url

    except Exception as e:
        logger.error(f"Error getting public URL: {e}", exc_info=True)
        return None


def list_files_for_group(group_id: str) -> list:
    """
    List all files for a specific group.

    Args:
        group_id: The group ID

    Returns:
        List of file objects or empty list on error
    """
    supabase = get_supabase_client()

    if not supabase:
        logger.error("Supabase client not initialized")
        return []

    try:
        # List files in the group's folder
        response = supabase.storage.from_(BUCKET_NAME).list(group_id)

        return response or []

    except Exception as e:
        logger.error(f"Error listing files for group {group_id}: {e}", exc_info=True)
        return []


def get_storage_info() -> dict:
    """
    Get information about the storage bucket.

    Returns:
        Dictionary with storage info or error
    """
    supabase = get_supabase_client()

    if not supabase:
        return {"error": "Supabase client not initialized"}

    try:
        # Get bucket info
        bucket_info = supabase.storage.get_bucket(BUCKET_NAME)

        return {
            "bucket_name": BUCKET_NAME,
            "bucket_info": bucket_info,
            "status": "available"
        }

    except Exception as e:
        logger.error(f"Error getting storage info: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "unavailable"
        }
