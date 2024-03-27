import pika
import json
import time
import sys

def callback(ch, method, properties, body):
    body = body.decode()
    data = json.loads(body)
    print(f"Received prediction results: {data}")

def connect_to_rabbitmq():
    while True:
        try:
            rabbitmq_url = 'amqp://guest:guest@localhost:5672/'
            return pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        except pika.exceptions.AMQPConnectionError:
            print("Failed to connect to RabbitMQ, retrying...", file=sys.stderr)
            time.sleep(5)

if __name__ == "__main__":
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue='predict_results')

    channel.basic_consume(queue='predict_results', on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for prediction results. To exit press CTRL+C')
    channel.start_consuming()
