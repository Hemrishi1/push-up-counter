from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models.user import User
from app.models.goal import Goal
from app.schemas.user import UserCreate, User as UserSchema
from app.schemas.auth import Token
from app.core.security import get_password_hash, verify_password, create_access_token, create_refresh_token
from app.core.config import settings
from jose import jwt, JWTError
from pydantic import BaseModel
from beanie import PydanticObjectId

router = APIRouter()

@router.post("/register", response_model=UserSchema)
async def register(user_in: UserCreate):
    existing = await User.find_one(User.email == user_in.email)
    if existing:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    user = User(
        email=user_in.email,
        password=get_password_hash(user_in.password),
        name=user_in.name,
    )
    await user.insert()

    # Create default goals for the user
    goal = Goal(user_id=user.id)
    await goal.insert()

    return UserSchema.from_beanie(user)

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await User.find_one(User.email == form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    return {
        "access_token": create_access_token(str(user.id)),
        "refresh_token": create_refresh_token(str(user.id)),
        "token_type": "bearer",
    }

class RefreshTokenRequest(BaseModel):
    refresh_token: str

@router.post("/refresh", response_model=Token)
async def refresh_token(request: RefreshTokenRequest):
    try:
        payload = jwt.decode(
            request.refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=403, detail="Not a refresh token")
        user_id_str = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
        
    try:
        user_id = PydanticObjectId(user_id_str)
    except:
        raise HTTPException(status_code=400, detail="Invalid token subject")

    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {
        "access_token": create_access_token(str(user.id)),
        "refresh_token": create_refresh_token(str(user.id)),
        "token_type": "bearer",
    }
