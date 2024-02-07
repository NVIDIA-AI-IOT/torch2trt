FROM nvcr.io/nvidia/tensorrt:19.08-py3

# optional python dependecies for tensorrt
RUN /opt/tensorrt/python/python_setup.sh

# system wide dependecies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    locales \
    libsm6 libxext6 libxrender-dev \
    libprotobuf* protobuf-compiler ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# locale fixes
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# miniconda for libtorch installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
RUN export PATH="$HOME/miniconda/bin:$PATH"

# additional dependecies for miniconda
RUN source $HOME/miniconda/bin/activate && \
    conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

# pytorch from sources
RUN git clone --recursive https://github.com/pytorch/pytorch
RUN cd pytorch
RUN source $HOME/miniconda/bin/activate && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# additional pytorch dependencies
RUN pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

# pytorch python module
WORKDIR /workspace/pytorch
RUN python setup.py install

# torch2trt with plugins
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd torch2trt && python setup.py install --plugins

WORKDIR /workspace
