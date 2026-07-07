from beanie import Document, PydanticObjectId
from pydantic import Field
from datetime import datetime

class Workout(Document):
    user_id: PydanticObjectId
    pushups: int = 0
    duration: int = 0 # in seconds
    calories: float = 0.0
    accuracy: float = 0.0
    average_speed: float = 0.0 # pushups per minute
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "workouts"
        indexes = [
            "user_id",
            "-created_at"
        ]
