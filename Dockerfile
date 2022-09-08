# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV TZ Asia/Seoul
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        ## Python
        python3-pip \
	    python3-setuptools \
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*



# Make working directories
# RUN  mkdir -p  /news-classfication
RUN git clone https://github.com/ady95/bert-for-korean-spacing.git
WORKDIR  /bert-for-korean-spacing

# Copy application requirements file to the created working directory
# COPY requirements.txt .
# COPY . .

# Install application dependencies from the requirements file
RUN pip3 install -U pip && pip3 install setuptools
RUN pip3 install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN pip3 install -r requirements.txt
    

# Run the uvicorn application
# CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
