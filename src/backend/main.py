from fastapi import FAstAPI
from fastapi.middleware.cors import CORSMiddleware
from recordings import router as recordings_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[" http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["8"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

app.include_router(recordings_router)

