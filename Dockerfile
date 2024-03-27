FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt update && \
    apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 && \
    apt upgrade --no-install-recommends -y openssl tar

WORKDIR /usr/src/flood

ENV PYTHONUNBUFFERED=1 

COPY . /usr/src/flood/  

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel && \
    pip install --no-cache -e ".[export]" comet pycocotools pytest-cov && \
    pip install --no-cache paddlepaddle==2.4.2 x2paddle && \
    pip install --no-cache numpy==1.23.5 && \
    pip install --no-cache lapx==0.5.2 && \
    pip install --no-cache py-cpuinfo && \
    pip install --no-cache pika && \
    pip install --no-cache pillow==10.2.0 && \
    pip install --no-cache torchvision==0.17.1 && \
    pip install --no-cache opencv-python==4.9.0.80 && \
    pip install --no-cache requests==2.31.0
    
# Remove exported models (if any)
RUN rm -rf tmp

# Set environment variables
ENV OMP_NUM_THREADS=1
ENV MKL_THREADING_LAYER=GNU

# Make port 80 available to the world outside this container
EXPOSE 80

# Define the directory where the store folder will be mounted
VOLUME /usr/src/flood/images

# Run predict.py when the container launches
CMD ["python", "predict.py"]
