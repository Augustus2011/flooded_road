import pika
import json
import time
import sys
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision
from torchvision.transforms import transforms
from torch.nn.functional import sigmoid,softmax



def read_from_url(url: str) -> Image:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image.convert("RGB")
        else:
            print(f"Failed to read an image from URL: {url}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Failed to read an image from URL: {url}. Error: {e}", file=sys.stderr)
        return None


def predict(url: str) -> dict:
    try:
        img = read_from_url(url=url)
        if img is None:
            return None
        
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 3)
        trans = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        model.load_state_dict(torch.load("/Users/kunkerdthaisong/cils/flooded_road/training_logs/3/exp03_best.pt",map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            start_time = time.time()
            img = img.resize((224, 224))
            img = trans(img)
            img = img.unsqueeze(0)
            out = model(img)
            end_time = time.time()
            
            cls = out.argmax(dim=-1).item()
            a=softmax(out.squeeze()).numpy()
            l=[float(i) for i in a]
            data = {"url": url, "result": cls,"probs":l, "time_predict": end_time - start_time}
            return data
    except Exception as e:
        print(f"Prediction failed for URL: {url}. Error: {e}", file=sys.stderr)
        return None


def publish_results(channel, data):
    channel.basic_publish(exchange='', routing_key='predict_results', body=json.dumps(data))
    print(" [x] Published results to 'predict_results' queue", file=sys.stderr, flush=True)


def callback(ch, method, properties, body):
    try:
        data = json.loads(body.decode())
        image_url = data.get("image_path_or_url")
        if image_url:
            print(f" [x] Received image URL: {image_url}")
            data = predict(image_url)
            if data:
                publish_results(ch, data)
        else:
            print(" [!] Invalid message format. Missing 'image_path_or_url' key.", file=sys.stderr)
    except json.JSONDecodeError:
        print(" [!] Failed to decode JSON message.", file=sys.stderr)



def connect_to_rabbitmq():
    while True:
        try:
            rabbitmq_url = 'amqp://guest:guest@localhost:5672/'
            return pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        except pika.exceptions.AMQPConnectionError:
            print("Failed to connect to RabbitMQ, retrying...", file=sys.stderr, flush=True)
            time.sleep(5)


if __name__ == "__main__":
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue='img_paths')
    channel.queue_declare(queue='predict_results')

    channel.basic_consume(queue='img_paths', on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for image paths. To exit press CTRL+C', file=sys.stderr, flush=True)
    channel.start_consuming()
