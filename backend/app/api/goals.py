from fastapi import APIRouter, Depends, HTTPException
from app.models.goal import Goal
from app.schemas.goal import Goal as GoalSchema, GoalUpdate
from app.auth.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.get("/", response_model=GoalSchema)
async def get_goals(
    current_user: User = Depends(get_current_user)
):
    goal = await Goal.find_one(Goal.user_id == current_user.id)
    if not goal:
        goal = Goal(user_id=current_user.id)
        await goal.insert()
    return goal

@router.put("/", response_model=GoalSchema)
async def update_goals(
    goal_in: GoalUpdate,
    current_user: User = Depends(get_current_user)
):
    goal = await Goal.find_one(Goal.user_id == current_user.id)
    if not goal:
        goal = Goal(user_id=current_user.id)
        
    if goal_in.daily_goal is not None:
        goal.daily_goal = goal_in.daily_goal
    if goal_in.weekly_goal is not None:
        goal.weekly_goal = goal_in.weekly_goal
    if goal_in.monthly_goal is not None:
        goal.monthly_goal = goal_in.monthly_goal
        
    await goal.save()
    return goal
