import requests
from PIL import Image
from io import BytesIO


#read image from url
def read_from_url(url:str)->Image:

    response = requests.get(url)
    
    if response.status_code == 200:

        image = Image.open(BytesIO(response.content)) #got PIL.Image object
        return image
    else:
        print(f"Failed to download image from URL: {url}")



#url = "https://example.com/image.jpg"
#save_path = "image.jpg"
#read_and_save_image(url, save_path)
