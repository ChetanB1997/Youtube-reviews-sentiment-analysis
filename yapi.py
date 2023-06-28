from fastapi import FastAPI
import requests
import uvicorn
from fastapi.responses import FileResponse
import os 

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to our channel. please like, share and subscribe"}


@app.post("/process_youtube_link")
async def process_youtube_link(link: str):
    file_path = "data/url.csv"  # Path to the file where you want to save the URL
    with open(file_path, "w") as file:
        file.write(link)
    return {"message": "Link processed successfully"}
    
@app.get("/image")
async def image():
    return FileResponse('result/saved_image.jpg')

if __name__ == '__main__':
    uvicorn.run()


