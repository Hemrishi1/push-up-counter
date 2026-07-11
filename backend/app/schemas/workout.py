from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class WorkoutBase(BaseModel):
    pushups: int
    duration: int
    calories: float
    accuracy: float
    average_speed: float

class WorkoutCreate(WorkoutBase):
    pass

class WorkoutInDBBase(WorkoutBase):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class Workout(WorkoutInDBBase):
    @classmethod
    def from_beanie(cls, workout_doc):
        return cls(
            id=str(workout_doc.id),
            user_id=str(workout_doc.user_id),
            pushups=workout_doc.pushups,
            duration=workout_doc.duration,
            calories=workout_doc.calories,
            accuracy=workout_doc.accuracy,
            average_speed=workout_doc.average_speed,
            created_at=workout_doc.created_at,
        )
