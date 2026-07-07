from fastapi import APIRouter, Depends
from datetime import datetime, timedelta, timezone
from app.models.workout import Workout
from app.auth.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.get("/dashboard")
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user)
):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Today's stats
    today_workouts = await Workout.find(
        Workout.user_id == current_user.id,
        Workout.created_at >= today
    ).to_list()
    
    today_pushups = sum(w.pushups for w in today_workouts)
    today_calories = sum(w.calories for w in today_workouts)
    today_duration = sum(w.duration for w in today_workouts)
    
    # Total stats
    all_workouts = await Workout.find(Workout.user_id == current_user.id).to_list()
    total_pushups = sum(w.pushups for w in all_workouts)
    total_workouts = len(all_workouts)
    
    # Calculate streak (simplified version)
    streak = 1 if today_pushups > 0 else 0
    
    return {
        "today_pushups": today_pushups,
        "today_calories": today_calories,
        "today_duration": today_duration,
        "total_pushups": total_pushups,
        "total_workouts": total_workouts,
        "streak": streak
    }

@router.get("/weekly")
async def get_weekly_stats(
    current_user: User = Depends(get_current_user)
):
    last_week = datetime.now(timezone.utc) - timedelta(days=7)
    workouts = await Workout.find(
        Workout.user_id == current_user.id,
        Workout.created_at >= last_week
    ).to_list()
    
    days = {}
    for i in range(7):
        d = (datetime.now() - timedelta(days=i)).date()
        days[d.strftime("%a")] = 0
        
    for w in workouts:
        day_str = w.created_at.strftime("%a")
        if day_str in days:
            days[day_str] += w.pushups
            
    # Format for charts
    chart_data = [{"name": k, "pushups": v} for k, v in reversed(days.items())]
    return chart_data
