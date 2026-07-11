from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.core.config import settings
from app.models.user import User
from app.models.workout import Workout
from app.models.goal import Goal
from app.models.achievement import Achievement

async def init_db():
    client = AsyncIOMotorClient(settings.MONGODB_URL)

    # Extract db name from URL or fall back to "pushup_db"
    db_name = settings.MONGODB_URL.rsplit("/", 1)[-1].split("?")[0] or "pushup_db"

    await init_beanie(
        database=client[db_name],
        document_models=[
            User,
            Workout,
            Goal,
            Achievement
        ]
    )
