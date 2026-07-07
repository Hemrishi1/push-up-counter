from pydantic import BaseModel
from datetime import datetime

class AchievementBase(BaseModel):
    title: str
    description: str

class AchievementCreate(AchievementBase):
    pass

class AchievementInDBBase(AchievementBase):
    id: str
    user_id: str
    earned_at: datetime

    class Config:
        from_attributes = True

class Achievement(AchievementInDBBase):
    pass
