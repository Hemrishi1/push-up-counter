from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.core.config import settings
from app.models.user import User
from app.models.workout import Workout
from app.models.goal import Goal
from app.models.achievement import Achievement

async def init_db():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    
    # Initialize Beanie with the client and document models
    await init_beanie(
        database=client.get_default_database(),
        document_models=[
            User,
            Workout,
            Goal,
            Achievement
        ]
    )

def get_db():
    """
    With Beanie, models are connected globally so we don't necessarily 
    need a session dependency like SQLAlchemy. But we'll keep the 
    structure for backward compatibility if needed, or endpoints 
    can just omit this dependency.
    """
    pass
