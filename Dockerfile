FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y

# Install useful tools
RUN apt-get install -y \ 
 wget

# Install Python3.10
RUN apt-get install -y \
 python3.10 \
 python3.10-dev

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

# Install OpenCV2
RUN apt-get install -y python3-opencv

# Install the rest of the requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --no-deps

RUN pip install regex

# Change workdir
WORKDIR /app

# Copy application
COPY . /app

# Run application
CMD ["python3.10", "ucf-crime/ucf_encoding.py"]