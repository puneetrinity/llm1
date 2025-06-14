# auth.py - Authentication Service
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger(__name__)

class AuthService:
    """Production authentication service with API keys and sessions"""
    
    def __init__(self, settings):
        self.settings = settings
        self.api_keys = {}
        self.sessions = {}
        self.failed_attempts = {}  # For rate limiting
        self.initialized = False
    
    def initialize(self):
        """Initialize the authentication service"""
        try:
            # Add default API key
            self._add_default_api_key()
            
            # Load additional API keys if configured
            self._load_api_keys()
            
            self.initialized = True
            logger.info("✅ Authentication service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize auth service: {e}")
            raise
    
    def _add_default_api_key(self):
        """Add the default API key"""
        api_key = self.settings.DEFAULT_API_KEY
        
        self.api_keys[api_key] = {
            "user_id": "admin",
            "name": "Default Admin",
            "permissions": ["read", "write", "admin"],
            "created_at": datetime.now(),
            "last_used": None,
            "active": True,
            "rate_limit": self.settings.DEFAULT_RATE_LIMIT,
            "key_type": "admin"
        }
        
        logger.info(f"🔑 Default API key added: {api_key[:8]}...")
    
    def _load_api_keys(self):
        """Load additional API keys from environment or file"""
        # This could be extended to load from a database or file
        # For now, we just use the default key
        pass
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return user information"""
        if not api_key or not self.initialized:
            return None
        
        # Check if key exists and is active
        key_info = self.api_keys.get(api_key)
        if not key_info or not key_info.get("active", False):
            self._log_failed_attempt(api_key)
            return None
        
        # Update last used timestamp
        key_info["last_used"] = datetime.now()
        
        # Return user information (without sensitive data)
        return {
            "user_id": key_info["user_id"],
            "name": key_info["name"],
            "permissions": key_info["permissions"],
            "rate_limit": key_info.get("rate_limit", self.settings.DEFAULT_RATE_LIMIT),
            "key_type": key_info.get("key_type", "user"),
            "api_key_id": hashlib.sha256(api_key.encode()).hexdigest()[:8]
        }
    
    def create_session(self, user_id: str, expires_hours: int = None) -> str:
        """Create a new session token"""
        if expires_hours is None:
            expires_hours = self.settings.SESSION_TIMEOUT_HOURS
        
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        # Clean up old sessions for this user
        self._cleanup_user_sessions(user_id)
        
        self.sessions[session_token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "last_accessed": datetime.now(),
            "active": True
        }
        
        logger.info(f"🎫 Session created for user: {user_id} (expires: {expires_at})")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate a session token"""
        if not session_token or session_token not in self.sessions:
            return None
        
        session = self.sessions[session_token]
        
        # Check if session is active and not expired
        if not session.get("active") or session["expires_at"] < datetime.now():
            self.revoke_session(session_token)
            return None
        
        # Update last accessed time
        session["last_accessed"] = datetime.now()
        
        # Get user info from API key store
        user_id = session["user_id"]
        for api_key, key_info in self.api_keys.items():
            if key_info["user_id"] == user_id and key_info["active"]:
                return {
                    "user_id": key_info["user_id"],
                    "name": key_info["name"],
                    "permissions": key_info["permissions"],
                    "session_token": session_token[:8] + "...",
                    "key_type": key_info.get("key_type", "user")
                }
        
        return None
    
    def revoke_session(self, session_token: str) -> bool:
        """Revoke a session token"""
        if session_token in self.sessions:
            del self.sessions[session_token]
            logger.info(f"🗑️ Session revoked: {session_token[:8]}...")
            return True
        return False
    
    def create_api_key(
        self, 
        user_id: str, 
        name: str, 
        permissions: List[str] = None,
        key_type: str = "user"
    ) -> str:
        """Create a new API key"""
        if permissions is None:
            permissions = ["read"]
        
        # Generate secure API key
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "name": name,
            "permissions": permissions,
            "created_at": datetime.now(),
            "last_used": None,
            "active": True,
            "rate_limit": self.settings.DEFAULT_RATE_LIMIT,
            "key_type": key_type
        }
        
        logger.info(f"🆕 API key created for {user_id}: {api_key[:8]}...")
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info(f"🚫 API key revoked: {api_key[:8]}...")
            return True
        return False
    
    def list_api_keys(self, user_id: str = None) -> List[Dict[str, Any]]:
        """List API keys (optionally filtered by user)"""
        keys = []
        for api_key, info in self.api_keys.items():
            if user_id is None or info["user_id"] == user_id:
                key_data = info.copy()
                key_data["api_key"] = api_key[:8] + "..." + api_key[-4:]
                keys.append(key_data)
        return keys
    
    def has_permission(self, user_info: Dict[str, Any], permission: str) -> bool:
        """Check if user has specific permission"""
        user_permissions = user_info.get("permissions", [])
        return permission in user_permissions or "admin" in user_permissions
    
    def _cleanup_user_sessions(self, user_id: str):
        """Clean up old sessions for a user"""
        user_sessions = [
            token for token, session in self.sessions.items()
            if session["user_id"] == user_id
        ]
        
        # Sort by creation time and keep only the latest ones
        user_sessions.sort(
            key=lambda token: self.sessions[token]["created_at"],
            reverse=True
        )
        
        # Remove excess sessions
        max_sessions = self.settings.MAX_SESSIONS_PER_USER
        for token in user_sessions[max_sessions:]:
            del self.sessions[token]
        
        if len(user_sessions) > max_sessions:
            logger.info(f"🧹 Cleaned up {len(user_sessions) - max_sessions} old sessions for {user_id}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_tokens = [
            token for token, session in self.sessions.items()
            if session["expires_at"] < current_time
        ]
        
        for token in expired_tokens:
            del self.sessions[token]
        
        if expired_tokens:
            logger.info(f"🧹 Cleaned up {len(expired_tokens)} expired sessions")
        
        return len(expired_tokens)
    
    def _log_failed_attempt(self, api_key: str):
        """Log failed authentication attempt"""
        key_hash = hashlib.sha256(api_key.encode() if api_key else b"").hexdigest()[:8]
        
        if key_hash not in self.failed_attempts:
            self.failed_attempts[key_hash] = {
                "count": 0,
                "first_attempt": datetime.now(),
                "last_attempt": datetime.now()
            }
        
        self.failed_attempts[key_hash]["count"] += 1
        self.failed_attempts[key_hash]["last_attempt"] = datetime.now()
        
        logger.warning(f"🚫 Failed authentication attempt: {key_hash}")
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        active_sessions = len([
            s for s in self.sessions.values()
            if s["active"] and s["expires_at"] > datetime.now()
        ])
        
        active_keys = len([
            k for k in self.api_keys.values()
            if k["active"]
        ])
        
        return {
            "active_sessions": active_sessions,
            "total_sessions": len(self.sessions),
            "active_api_keys": active_keys,
            "total_api_keys": len(self.api_keys),
            "failed_attempts": len(self.failed_attempts),
            "initialized": self.initialized
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export safe configuration for frontend"""
        return {
            "auth_enabled": self.settings.ENABLE_AUTH,
            "environment": self.settings.ENVIRONMENT,
            "api_key_header": self.settings.API_KEY_HEADER,
            "session_timeout_hours": self.settings.SESSION_TIMEOUT_HOURS,
            "max_sessions_per_user": self.settings.MAX_SESSIONS_PER_USER,
            "rate_limit_default": self.settings.DEFAULT_RATE_LIMIT
        }
