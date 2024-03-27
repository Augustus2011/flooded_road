# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install pika for RabbitMQ communication
RUN pip install pika

ENV PYTHONUNBUFFERED=1 

# Copy the producer script into the container
COPY producer.py .

# Define a volume for the image directory
VOLUME /usr/src/app/image

# Command to run the producer script
CMD ["python", "producer.py"]
