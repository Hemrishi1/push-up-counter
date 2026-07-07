from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, users, workouts, analytics, goals, achievements
from app.core.config import settings
from app.db.database import init_db

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Set all CORS enabled origins
if settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.cors_origins_list],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
async def start_db():
    await init_db()

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/user", tags=["user"])
app.include_router(workouts.router, prefix="/api/workout", tags=["workout"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(goals.router, prefix="/api/goals", tags=["goals"])
app.include_router(achievements.router, prefix="/api/achievements", tags=["achievements"])

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
