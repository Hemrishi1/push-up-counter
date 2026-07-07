# AI Push-up Counter & Fitness Tracker

A full-stack SaaS application that tracks push-up workouts using AI (MediaPipe & OpenCV) entirely in the browser using WebSockets to communicate with a FastAPI backend.

## Architecture

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Shadcn UI, Framer Motion, Recharts
- **Backend**: Python, FastAPI, Beanie, Motor, OpenCV, MediaPipe
- **Database**: MongoDB (Atlas)
- **Deployment**: Vercel (Frontend), Render (Backend), Docker

## Folder Structure

```
.
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── ai/               # AI Engine (pushup_counter.py wrapped by pushup_service.py)
│   │   ├── api/              # REST & WebSocket API Routes
│   │   ├── auth/             # JWT Authentication
│   │   ├── core/             # Settings and Security
│   │   ├── db/               # MongoDB Connection Setup
│   │   ├── models/           # Database Models
│   │   └── schemas/          # Pydantic Schemas
│   ├── tests/                # Pytest Suite
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # React Application
│   ├── src/
│   │   ├── components/       # Reusable UI Components
│   │   ├── context/          # React Context (Auth)
│   │   ├── layouts/          # Layout Components
│   │   ├── pages/            # Application Pages
│   │   ├── services/         # API Clients (Axios)
│   │   ├── App.tsx           # Router
│   │   └── main.tsx          # Entry Point
│   ├── Dockerfile
│   └── package.json
└── docker-compose.yml        # Local execution
```

## Installation & Local Development

### Prerequisites
- Docker and Docker Compose
- Node.js (v20+)
- Python (3.11+)

### Using Docker (Recommended)
1. Clone the repository
2. Run `docker-compose up --build`
3. Access Frontend at `http://localhost:5173`
4. Access Backend API at `http://localhost:8000/api/docs`

### Manual Setup
**Backend:**
1. `cd backend`
2. `pip install -r requirements.txt`
3. Set up a local MongoDB database and configure `.env` (copy from `.env.example`)
4. Run `uvicorn app.main:app --reload`

**Frontend:**
1. `cd frontend`
2. `npm install`
3. Copy `.env.example` to `.env` (if applicable) and point `VITE_API_URL` to `http://localhost:8000/api`
4. Run `npm run dev`

## Deployment

- **Frontend -> Vercel**: Connect the repository to Vercel, set root directory to `frontend`. Ensure `VITE_API_URL` is set to your Render backend URL.
- **Backend -> Render**: Connect the repository to Render, choose Web Service, set root directory to `backend`, set start command to `uvicorn app.main:app --host 0.0.0.0 --port 10000`. Add your `MONGODB_URL` environment variable.
- **Database -> Atlas**: Create a MongoDB database on MongoDB Atlas and provide the connection URL to the backend.

## Integrating AI Engine

The original `pushup_counter.py` was refactored slightly to isolate the `poseDetector` class. 
The main detection loop is now in `backend/app/ai/pushup_service.py` (`process_frame`), which receives base64 encoded frames from the frontend via WebSockets, processes them using `poseDetector`, calculates stats, and sends the annotated frame back in real-time. This ensures that the webcam capture logic remains in the client browser, while the heavy AI processing occurs securely on the server.
