FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
LABEL maintainer "Sangin Lee <silee@rcv.sejong.ac.kr>"

ENV LANG C.UTF-8

ARG PYTHON_VERSION=
ARG CONDA_ENV_NAME=

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    apt-utils \
    build-essential \
    sudo \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ssh \
    wget \
    unzip \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

RUN conda update -y conda && \
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}

ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
RUN echo $PATH

COPY requirements.txt /tmp/requirements.txt
RUN source activate ${CONDA_ENV_NAME} && pip install --no-cache-dir -r tmp/requirements.txt

SHELL ["/bin/bash", "-c"]

RUN adduser $USER_NAME -u $UID --quiet --gecos "" --disabled-password && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

USER $USER_NAME

SHELL ["/bin/bash", "-c"]

RUN echo "source activate ${CONDA_ENV_NAME}" > ~/.bashrc

CMD ["/bin/bash"]
ENV CONDA_DEFAULT_ENV ${CONDA_ENV_NAME}

RUN source activate ${CONDA_ENV_NAME} && pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
