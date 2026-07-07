from fastapi import APIRouter, Depends, HTTPException
from app.models.user import User
from app.schemas.user import User as UserSchema, UserUpdate
from app.auth.dependencies import get_current_user
from app.core.security import get_password_hash

router = APIRouter()

@router.get("/profile", response_model=UserSchema)
async def read_user_profile(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/profile", response_model=UserSchema)
async def update_user_profile(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    if user_in.name is not None:
        current_user.name = user_in.name
    if user_in.email is not None:
        user = await User.find_one(User.email == user_in.email)
        if user and user.id != current_user.id:
            raise HTTPException(status_code=400, detail="Email already registered")
        current_user.email = user_in.email
    if user_in.avatar is not None:
        current_user.avatar = user_in.avatar
    if user_in.password is not None:
        current_user.password = get_password_hash(user_in.password)
        
    await current_user.save()
    return current_user
