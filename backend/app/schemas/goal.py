from pydantic import BaseModel
from typing import Optional

class GoalBase(BaseModel):
    daily_goal: int = 50
    weekly_goal: int = 300
    monthly_goal: int = 1000

class GoalCreate(GoalBase):
    pass

class GoalUpdate(GoalBase):
    daily_goal: Optional[int] = None
    weekly_goal: Optional[int] = None
    monthly_goal: Optional[int] = None

class GoalInDBBase(GoalBase):
    id: str
    user_id: str

    class Config:
        from_attributes = True

class Goal(GoalInDBBase):
    pass
