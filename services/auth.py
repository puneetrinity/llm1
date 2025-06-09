# services/auth.py - Authentication Service
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from config import Settings

class AuthService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        
        # In-memory API key store (in production, use a proper database)
        self.api_keys = {
            self.settings.DEFAULT_API_KEY: {
                "user_id": "default_user",
                "name": "Default API Key",
                "permissions": ["read", "write"],
                "rate_limit": 100,  # requests per minute
                "created_at": datetime.now(),
                "last_used": None,
                "active": True
            }
        }
        
        # User sessions (for WebSocket authentication)
        self.sessions = {}
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information"""
        
        if not api_key or not self.settings.ENABLE_AUTH:
            # If auth is disabled, return default user
            return {
                "user_id": "anonymous",
                "permissions": ["read", "write"],
                "rate_limit": self.settings.DEFAULT_RATE_LIMIT
            }
        
        # Check if API key exists and is active
        key_info = self.api_keys.get(api_key)
        if not key_info or not key_info.get("active", False):
            return None
        
        # Update last used timestamp
        key_info["last_used"] = datetime.now()
        
        return {
            "user_id": key_info["user_id"],
            "name": key_info.get("name", "Unknown"),
            "permissions": key_info.get("permissions", ["read"]),
            "rate_limit": key_info.get("rate_limit", self.settings.DEFAULT_RATE_LIMIT),
            "api_key": api_key[:8] + "..."  # Truncated for logging
        }
    
    def create_api_key(
        self, 
        user_id: str, 
        name: str, 
        permissions: list = None,
        rate_limit: int = None
    ) -> str:
        """Create a new API key"""
        
        if permissions is None:
            permissions = ["read"]
        
        if rate_limit is None:
            rate_limit = self.settings.DEFAULT_RATE_LIMIT
        
        # Generate secure API key
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "created_at": datetime.now(),
            "last_used": None,
            "active": True
        }
        
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            return True
        
        return False
    
    def list_api_keys(self, user_id: str = None) -> list:
        """List API keys (optionally filtered by user)"""
        
        keys = []
        for api_key, info in self.api_keys.items():
            if user_id is None or info["user_id"] == user_id:
                # Don't expose the full API key
                key_info = info.copy()
                key_info["api_key"] = api_key[:8] + "..." + api_key[-4:]
                keys.append(key_info)
        
        return keys
    
    def has_permission(self, user_info: Dict[str, Any], permission: str) -> bool:
        """Check if user has specific permission"""
        
        user_permissions = user_info.get("permissions", [])
        return permission in user_permissions or "admin" in user_permissions
    
    def create_session(self, user_id: str, expires_in_hours: int = 24) -> str:
        """Create a session token for WebSocket authentication"""
        
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        self.sessions[session_token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "active": True
        }
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token"""
        
        session_info = self.sessions.get(session_token)
        if not session_info or not session_info.get("active", False):
            return None
        
        # Check if session has expired
        if datetime.now() > session_info["expires_at"]:
            session_info["active"] = False
            return None
        
        return {
            "user_id": session_info["user_id"],
            "session_token": session_token[:8] + "...",
            "expires_at": session_info["expires_at"]
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        
        current_time = datetime.now()
        expired_sessions = [
            token for token, info in self.sessions.items()
            if current_time > info["expires_at"]
        ]
        
        for token in expired_sessions:
            del self.sessions[token]
        
        return len(expired_sessions)
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user"""
        
        user_keys = [
            info for info in self.api_keys.values()
            if info["user_id"] == user_id
        ]
        
        active_keys = sum(1 for key in user_keys if key.get("active", False))
        last_used = max(
            (key["last_used"] for key in user_keys if key["last_used"]),
            default=None
        )
        
        return {
            "user_id": user_id,
            "total_api_keys": len(user_keys),
            "active_api_keys": active_keys,
            "last_used": last_used,
            "permissions": user_keys[0].get("permissions", []) if user_keys else []
        }
