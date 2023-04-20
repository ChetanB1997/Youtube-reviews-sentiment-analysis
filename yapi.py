from fastapi import FastAPI
import requests
    
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to our channel. please like, share and subscribe"}


@app.post("/process_youtube_link")
async def process_youtube_link(link: str):
    return {"message": "Link processed successfully"}




file_path = "F:\Project\youtube comment sentiment analysis\dog_image.jpg"


@app.get("/image")
async def image():
    return FileResponse(file_path)
