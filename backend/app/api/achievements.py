from fastapi import APIRouter, Depends
from typing import List
from app.models.achievement import Achievement
from app.schemas.achievement import Achievement as AchievementSchema
from app.auth.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.get("/", response_model=List[AchievementSchema])
async def get_achievements(
    current_user: User = Depends(get_current_user)
):
    achievements = await Achievement.find(Achievement.user_id == current_user.id).to_list()
    return achievements
