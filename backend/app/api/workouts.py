from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import List
from app.models.workout import Workout
from app.schemas.workout import Workout as WorkoutSchema, WorkoutCreate
from app.auth.dependencies import get_current_user
from app.models.user import User
import json

router = APIRouter()

@router.post("/end", response_model=WorkoutSchema)
async def end_workout(
    workout_in: WorkoutCreate,
    current_user: User = Depends(get_current_user)
):
    workout = Workout(
        user_id=current_user.id,
        pushups=workout_in.pushups,
        duration=workout_in.duration,
        calories=workout_in.calories,
        accuracy=workout_in.accuracy,
        average_speed=workout_in.average_speed
    )
    await workout.insert()
    return WorkoutSchema.from_beanie(workout)

@router.get("/history", response_model=List[WorkoutSchema])
async def get_workout_history(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    workouts = await Workout.find(Workout.user_id == current_user.id).sort(-Workout.created_at).skip(skip).limit(limit).to_list()
    return [WorkoutSchema.from_beanie(w) for w in workouts]

@router.get("/latest", response_model=WorkoutSchema)
async def get_latest_workout(
    current_user: User = Depends(get_current_user)
):
    workout = await Workout.find(Workout.user_id == current_user.id).sort(-Workout.created_at).first_or_none()
    if not workout:
        raise HTTPException(status_code=404, detail="No workouts found")
    return WorkoutSchema.from_beanie(workout)

from app.ai.pushup_service import ConnectionManager, process_frame

manager = ConnectionManager()

@router.websocket("/stream/{token}")
async def workout_stream(websocket: WebSocket, token: str):
    await manager.connect(websocket)
    try:
        session_state = {
            "count": 0,
            "dir": 0,
            "angle_history": [],
            "pTime": 0
        }
        
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "frame":
                    frame_data = message.get("data")
                    result = process_frame(frame_data, session_state)
                    await manager.send_personal_message(json.dumps(result), websocket)
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
