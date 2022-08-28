FROM susaha/damtl

RUN apt-get -y update

RUN apt-get -y update
RUN rm /var/lib/apt/lists/lock

RUN apt-get -y update
RUN apt-get install -y \
    net-tools \
    ssh-client \
    sshfs \
    apt-utils \
    wget \
    libopenmpi-dev \
    unzip \
    ffmpeg \
    tmux \
    psmisc \
    screen

ARG project_dir=/raid/gusingh/
WORKDIR $project_dir
ADD setup* $project_dir 

# commonly used
#RUN pip install --upgrade pip
#RUN pip install cython
#RUN pip install Pillow
#RUN pip install numpy
#RUN pip install scikit-image
#RUN pip install tqdm
#RUN pip install pyyaml
#RUN pip install matplotlib
#RUN pip install torch
#RUN pip install torchvision
#RUN pip install protobuf
#RUN pip install psutil
#RUN pip install tensorboardX
#RUN pip install gpustat
#RUN pip install opencv-python
#RUN pip install scikit-learn
#RUN pip install torch-encoding
#RUN pip install pandas
#RUN pip install imgaug
#RUN pip install tensorboard
#RUN pip install pycocotools
#RUN pip install h5py

# adventcvpr2019
#RUN pip install easydict

# tensorflow
# RUN pip install tensorflow tensorflow-gpu

# mti_simon
#RUN pip install termcolor
#RUN pip install imageio
#RUN pip install wrapt==1.10.0 --ignore-installed
## Added by gurkirt
RUN pip install tensorboard
RUN conda install av -c conda-forge 
RUN conda install ffmpeg=4.2 -c conda-forge 
RUN pip install simplejson
RUN pip install psutil
RUN pip install moviepy
RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
RUN python setup.py build develop
