from beanie import Document, PydanticObjectId

class Goal(Document):
    user_id: PydanticObjectId
    daily_goal: int = 50
    weekly_goal: int = 300
    monthly_goal: int = 1000

    class Settings:
        name = "goals"
        indexes = [
            "user_id"
        ]
