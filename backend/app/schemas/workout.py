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
    pass
