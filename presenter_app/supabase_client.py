import os
import logging
from supabase import create_client, Client

logger = logging.getLogger(__name__)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key) if url and key else None

if not supabase:
    logger.warning("Supabase client not initialized - missing SUPABASE_URL or SUPABASE_ANON_KEY")

def get_supabase_client() -> Client:
    return supabase

# --- Supabase CRUD operations for Group Portal ---

def create_group(group_name: str, project_title: str) -> dict:
    if not supabase:
        logger.error("Supabase client not initialized. Cannot create group.")
        return None
    try:
        response = supabase.table('groups').insert({
            "group_name": group_name,
            "project_title": project_title
        }).execute()
        logger.info(f"Group created: {response.data[0]['id'] if response.data else 'unknown'}")
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating group '{group_name}': {e}", exc_info=True)
        return None

def add_group_member(group_id: str, member_name: str) -> dict:
    if not supabase:
        logger.error("Supabase client not initialized. Cannot add group member.")
        return None
    try:
        response = supabase.table('group_members').insert({
            "group_id": group_id,
            "member_name": member_name
        }).execute()
        logger.info(f"Member '{member_name}' added to group {group_id}")
        return response.data[0]
    except Exception as e:
        logger.error(f"Error adding member '{member_name}' to group {group_id}: {e}", exc_info=True)
        return None

def add_group_document(group_id: str, document_title: str, file_path: str) -> dict:
    if not supabase:
        logger.error("Supabase client not initialized. Cannot add group document.")
        return None
    try:
        response = supabase.table('group_documents').insert({
            "group_id": group_id,
            "document_title": document_title,
            "file_path": file_path
        }).execute()
        logger.info(f"Document '{document_title}' added to group {group_id}")
        return response.data[0]
    except Exception as e:
        logger.error(f"Error adding document to group {group_id}: {e}", exc_info=True)
        return None

def get_groups() -> list:
    if not supabase:
        logger.error("Supabase client not initialized. Cannot get groups.")
        return []
    try:
        response = supabase.table('groups').select('*').execute()
        logger.info(f"Retrieved {len(response.data)} groups")
        return response.data
    except Exception as e:
        logger.error(f"Error getting groups: {e}", exc_info=True)
        return []

def get_group_details(group_id: str) -> dict:
    if not supabase:
        logger.error("Supabase client not initialized. Cannot get group details.")
        return None
    try:
        group_response = supabase.table('groups').select('*').eq('id', group_id).execute()
        group_data = group_response.data[0] if group_response.data else None

        if group_data:
            members_response = supabase.table('group_members').select('*').eq('group_id', group_id).execute()
            documents_response = supabase.table('group_documents').select('*').eq('group_id', group_id).execute()
            group_data['members'] = members_response.data
            group_data['documents'] = documents_response.data
            logger.info(f"Retrieved group {group_id} with {len(members_response.data)} members and {len(documents_response.data)} documents")
        return group_data
    except Exception as e:
        logger.error(f"Error getting group details for {group_id}: {e}", exc_info=True)
        return None

def delete_group(group_id: str) -> bool:
    if not supabase:
        logger.error("Supabase client not initialized. Cannot delete group.")
        return False
    try:
        # Delete associated documents first
        supabase.table('group_documents').delete().eq('group_id', group_id).execute()
        # Delete associated members
        supabase.table('group_members').delete().eq('group_id', group_id).execute()
        # Delete the group itself
        response = supabase.table('groups').delete().eq('id', group_id).execute()
        logger.info(f"Group {group_id} deleted successfully")
        return len(response.data) > 0
    except Exception as e:
        logger.error(f"Error deleting group {group_id}: {e}", exc_info=True)
        return False
