# Set the base as the nvidia-cuda Docker
FROM nvidia/cuda:8.0-devel

# Create directory for all of the files to go into and cd into it
WORKDIR /app

# Apt-get all needed dependencies
RUN apt-get update
RUN apt-get install -y git wget make gcc python python-pip build-essential curl \
		 cmake libreadline-dev git-core libqt4-dev libjpeg-dev \
		 libpng-dev ncurses-dev imagemagick libzmq3-dev gfortran \
		 unzip gnuplot gnuplot-x11 sudo vim libopencv-dev google-perftools \
		 libgoogle-perftools-dev ffmpeg
RUN apt-get install -y --no-install-recommends libhdf5-serial-dev liblmdb-dev
RUN echo "LD_PRELOAD=/usr/lib/libtcmalloc.so.4" | tee -a /etc/environment
ENV LD_PRELOAD "/usr/lib/libtcmalloc.so.4:$LD_PRELOAD"

# Install cuDNN and the dev files for cuDNN
WORKDIR /
COPY ./docker_files/cudnn-8.0-linux-x64-v5.1.tgz /
RUN tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
RUN cp cuda/include/cudnn*.h /usr/include/x86_64-linux-gnu/
RUN cp cuda/lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
RUN chmod a+r /usr/include/x86_64-linux-gnu/cudnn*.h /usr/lib/x86_64-linux-gnu//libcudn*

# Make links
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_static_v5.a libcudnn_stlib
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.5 libcudnn_so
RUN ln -sf /usr/include/x86_64-linux-gnu/cudnn_v5.h libcudnn

# Install anaconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

# create environment
RUN conda env create -n actionvlad python=3.8 pip

RUN conda init bash

RUN conda activate actionvlad


# Clone git repo
<<<<<<< Updated upstream
RUN git clone -b master https://github.com/aiporre/actionvlad.git /app/ActionVLAD --recursive
=======
RUN git clone -b main https://github.com/aiporre/actionvlad.git /app/ActionVLAD --recursive
RUN pip install -r /app/ActionVlad/requirements.txt
>>>>>>> Stashed changes
WORKDIR /app/ActionVLAD/

# install requirements
RUN pip install -r requirements.txt

#copy weights
COPY ./docker_files/models/kmeans-init/hmdb51/rgb_conv5_kmeans64.pkl /app/ActionVLAD/models/kmeans-init/hmdb51/
COPY ./docker_files/models/PreTrained/2-stream-pretrained/ucf/flow/split1.ckpt /app/ActionVLAD/models/PreTrained/2-stream-pretrained/ucf/flow/
COPY ./docker_files/models/PreTrained/2-stream-pretrained/ucf/flow/split2.ckpt /app/ActionVLAD/models/PreTrained/2-stream-pretrained/ucf/flow/
COPY ./docker_files/models/PreTrained/2-stream-pretrained/ucf/flow/split3.ckpt /app/ActionVLAD/models/PreTrained/2-stream-pretrained/ucf/flow/


COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/ucf/flow/split1.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/ucf/flow/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/ucf/flow/split2.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/ucf/flow/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/ucf/flow/split3.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/ucf/flow/

COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/ucf/rgb/split1.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/ucf/rgb/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/ucf/rgb/split2.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/ucf/rgb/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/ucf/rgb/split3.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/ucf/rgb/

COPY ./docker_files/models/PreTrained/imagenet-trained-CUHK/vgg_16_action_rgb_pretrain_uptoConv5.ckpt /app/ActionVLAD/models/PreTrained/imagenet-trained-CUHK/

#copy video to test
COPY ./docker_files/v_Archery_g01_c01.mp4 /app/ActionVLAD/demo/

# Remove the install files for cuDNN
WORKDIR /

WORKDIR /app/ActionVLAD/
