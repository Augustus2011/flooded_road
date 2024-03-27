import requests
from PIL import Image
from io import BytesIO


#read image from url
def read_from_url(url:str)->Image:

    response = requests.get(url)
    
    if response.status_code == 200:

        image = Image.open(BytesIO(response.content)) #got PIL.Image object
        return image.convert("RGB")
    else:
        print(f"Failed to download image from URL: {url}")




if __name__=="__main__":
    print(read_from_url("https://mhole.b.cils.cloud/api/v1/static/odessey/latest.png"))