import pika
import os
import time
import sys
import json

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

    # Construct the message with the image path or URL
    message = {"image_path_or_url": sys.argv[1]}
    
    # Convert the message to JSON and send it to the 'img_paths' queue
    channel.basic_publish(exchange='', routing_key='img_paths', body=json.dumps(message))
    
    # Wait for 100 seconds before exiting (optional)
    time.sleep(100)
