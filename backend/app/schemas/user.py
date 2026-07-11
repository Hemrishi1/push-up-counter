from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    avatar: Optional[str] = None
    password: Optional[str] = None

class User(BaseModel):
    id: str
    name: str
    email: EmailStr
    avatar: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @classmethod
    def from_beanie(cls, user_doc):
        return cls(
            id=str(user_doc.id),
            name=user_doc.name,
            email=user_doc.email,
            avatar=user_doc.avatar,
            created_at=user_doc.created_at,
            updated_at=user_doc.updated_at,
        )
