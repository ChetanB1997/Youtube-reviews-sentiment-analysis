from fastapi import FastAPI
import requests
    
from fastapi.responses import FileResponse

#app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Welcome to our channel. please like, share and subscribe"}


# @app.post("/process_youtube_link")
# async def process_youtube_link(link: str):
#     return {"message": "Link processed successfully"}




# file_path = "F:\Project\youtube comment sentiment analysis\dog_image.jpg"


# @app.get("/image")
# async def image():
#     return FileResponse(file_path)

#####
import requests
app = FastAPI()

BASE_URL = "http://localhost:8000"


def process_url(url: str):
    endpoint = BASE_URL + "/process_url"
    data = {"url": url}
    headers = {"Content-Type": "application/json"}
    response = requests.post(endpoint, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"message": "Failed to process URL"}


def main():
    url = input("Enter URL: ")
    result = process_url(url)
    print(result)


if __name__ == "__main__":
    main()
