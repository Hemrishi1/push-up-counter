from beanie import Document
from pydantic import EmailStr, Field
from datetime import datetime
from typing import Optional

class User(Document):
    name: str
    email: EmailStr
    password: str
    avatar: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "users"
        indexes = [
            "email",
            "name"
        ]
