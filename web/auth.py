"""
Authentication system for the Worm Annotation Tool.
Simple username/password auth with session tokens.
"""
import hashlib
import secrets
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass, asdict
from functools import wraps

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Configuration
AUTH_FILE = Path(__file__).parent / "users.json"
USER_PREFS_FILE = Path(__file__).parent / "user_preferences.json"
SESSION_EXPIRY_HOURS = 24


@dataclass
class UserPreferences:
    """Per-user preferences and state."""
    username: str
    last_project_path: str = ""
    last_folder_index: int = 0
    last_video_index: int = 0
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class User:
    username: str
    password_hash: str
    role: str = "annotator"  # admin, annotator, viewer
    created_at: str = ""
    last_login: str = ""
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class AuthManager:
    """Manages users and authentication."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, dict] = {}  # token -> {username, expires}
        self.user_prefs: Dict[str, UserPreferences] = {}  # username -> preferences
        self._load_users()
        self._load_user_prefs()
    
    def _load_users(self):
        """Load users from JSON file."""
        if AUTH_FILE.exists():
            try:
                with open(AUTH_FILE, 'r') as f:
                    data = json.load(f)
                    for username, user_data in data.get('users', {}).items():
                        self.users[username] = User.from_dict(user_data)
                print(f"Loaded {len(self.users)} users")
            except Exception as e:
                print(f"Error loading users: {e}")
        
        # Create default admin if no users exist
        if not self.users:
            self._create_default_admin()
    
    def _save_users(self):
        """Save users to JSON file."""
        data = {
            'users': {username: user.to_dict() for username, user in self.users.items()}
        }
        with open(AUTH_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_user_prefs(self):
        """Load user preferences from JSON file."""
        if USER_PREFS_FILE.exists():
            try:
                with open(USER_PREFS_FILE, 'r') as f:
                    data = json.load(f)
                    for username, prefs_data in data.items():
                        self.user_prefs[username] = UserPreferences.from_dict(prefs_data)
                print(f"Loaded preferences for {len(self.user_prefs)} users")
            except Exception as e:
                print(f"Error loading user preferences: {e}")
    
    def _save_user_prefs(self):
        """Save user preferences to JSON file."""
        data = {username: prefs.to_dict() for username, prefs in self.user_prefs.items()}
        with open(USER_PREFS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_user_prefs(self, username: str) -> UserPreferences:
        """Get preferences for a user, creating default if needed."""
        if username not in self.user_prefs:
            self.user_prefs[username] = UserPreferences(username=username)
        return self.user_prefs[username]
    
    def save_user_prefs(self, username: str, project_path: str = None, 
                        folder_index: int = None, video_index: int = None):
        """Update and save user preferences."""
        prefs = self.get_user_prefs(username)
        
        if project_path is not None:
            prefs.last_project_path = project_path
        if folder_index is not None:
            prefs.last_folder_index = folder_index
        if video_index is not None:
            prefs.last_video_index = video_index
        
        self._save_user_prefs()
    
    def _create_default_admin(self):
        """Create default admin account."""
        self.create_user("admin", "wormadmin123", role="admin")
        print("Created default admin user (username: admin, password: wormadmin123)")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password with salt."""
        salt = "worm_annotation_salt_2024"
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    
    def create_user(self, username: str, password: str, role: str = "annotator") -> bool:
        """Create a new user."""
        if username in self.users:
            return False
        
        self.users[username] = User(
            username=username,
            password_hash=self._hash_password(password),
            role=role,
            created_at=datetime.now().isoformat()
        )
        self._save_users()
        return True
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify username and password."""
        if username not in self.users:
            return False
        return self.users[username].password_hash == self._hash_password(password)
    
    def login(self, username: str, password: str) -> Optional[str]:
        """
        Attempt login and return session token if successful.
        """
        if not self.verify_password(username, password):
            return None
        
        # Update last login
        self.users[username].last_login = datetime.now().isoformat()
        self._save_users()
        
        # Create session token
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {
            'username': username,
            'expires': (datetime.now() + timedelta(hours=SESSION_EXPIRY_HOURS)).isoformat(),
            'role': self.users[username].role
        }
        
        return token
    
    def logout(self, token: str):
        """Invalidate a session token."""
        if token in self.sessions:
            del self.sessions[token]
    
    def validate_token(self, token: str) -> Optional[dict]:
        """
        Validate a session token.
        Returns session info if valid, None otherwise.
        """
        if token not in self.sessions:
            return None
        
        session = self.sessions[token]
        expires = datetime.fromisoformat(session['expires'])
        
        if datetime.now() > expires:
            del self.sessions[token]
            return None
        
        return session
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)
    
    def list_users(self) -> list:
        """List all users (without password hashes)."""
        return [
            {
                'username': u.username,
                'role': u.role,
                'created_at': u.created_at,
                'last_login': u.last_login
            }
            for u in self.users.values()
        ]
    
    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        if username not in self.users:
            return False
        if username == 'admin':
            return False  # Protect admin account
        
        del self.users[username]
        self._save_users()
        
        # Invalidate their sessions
        self.sessions = {
            k: v for k, v in self.sessions.items() 
            if v['username'] != username
        }
        return True
    
    def change_password(self, username: str, new_password: str) -> bool:
        """Change a user's password."""
        if username not in self.users:
            return False
        
        self.users[username].password_hash = self._hash_password(new_password)
        self._save_users()
        return True


# Global auth manager instance
auth_manager = AuthManager()


# FastAPI security
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[dict]:
    """
    Dependency to get current authenticated user.
    Returns None if not authenticated (for optional auth).
    """
    # Check Authorization header
    if credentials:
        session = auth_manager.validate_token(credentials.credentials)
        if session:
            return session
    
    # Check cookie as fallback
    token = request.cookies.get('auth_token')
    if token:
        session = auth_manager.validate_token(token)
        if session:
            return session
    
    # Check X-Auth-Token header (for JS clients)
    token = request.headers.get('X-Auth-Token')
    if token:
        session = auth_manager.validate_token(token)
        if session:
            return session
    
    return None


async def require_auth(
    user: Optional[dict] = Depends(get_current_user)
) -> dict:
    """Dependency that requires authentication."""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def require_admin(
    user: dict = Depends(require_auth)
) -> dict:
    """Dependency that requires admin role."""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
