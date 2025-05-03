import sys
sys.path.append('/Users/pepelopez/Documents/Programming/MoEtonomous')

from fastapi import FastAPI
from app.core.config import settings
from server.app.controller import pdf_controller
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",  # React dev server
    # "https://yourfrontenddomain.com",  # production frontend
]

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(pdf_controller.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies / auth headers
    allow_methods=["*"],               # Allow all HTTP methods
    allow_headers=["*"],               # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI MVC Example!"}