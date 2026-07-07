from beanie import Document, PydanticObjectId
from pydantic import Field
from datetime import datetime

class Achievement(Document):
    user_id: PydanticObjectId
    title: str
    description: str
    earned_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "achievements"
        indexes = [
            "user_id"
        ]
