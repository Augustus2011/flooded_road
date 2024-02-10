import requests
from PIL import Image
from io import BytesIO


#read image from url
def read_and_save_image(url, save_path):

    response = requests.get(url)
    
    if response.status_code == 200:

        image = Image.open(BytesIO(response.content))
        
        image.save(save_path)
        print(f"Image saved successfully at: {save_path}")
    else:
        print(f"Failed to download image from URL: {url}")

url = "https://example.com/image.jpg"
save_path = "image.jpg"

read_and_save_image(url, save_path)
