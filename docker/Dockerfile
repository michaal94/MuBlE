FROM ubuntu:20.04 as dev-base
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.7.4 conda-build pyyaml numpy ipython&& \
    /opt/conda/bin/conda clean -ya

RUN conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN apt-get update \
    && apt-get install -y \
	sudo \
	wget \ 
	libopenexr-dev \ 
	bzip2 \ 
	build-essential \ 
	zlib1g-dev \ 
	libxmu-dev \ 
	libxi-dev \
    libglib2.0-0 \ 
	libxxf86vm-dev \ 
	libfontconfig1 \ 
	libxrender1 \ 
	libgl1-mesa-glx \ 
	libosmesa6-dev \ 
	xz-utils

# RUN wget https://builder.blender.org/download/daily/blender-2.83.20-candidate+v283.92d3a152391a-linux.x86_64-release.tar.xz \
# 	&& tar -xvf blender-2.83.20-candidate+v283.92d3a152391a-linux.x86_64-release.tar.xz --strip-components=1 -C /bin \
# 	&& rm -rf blender-2.83.20-candidate+v283.92d3a152391a-linux.x86_64-release.tar.xz \
# 	&& rm -rf blender-2.83.20-candidate+v283.92d3a152391a-linux.x86_64-release \
#     && rm -rf /bin/2.83/python \
#     && ln -s /opt/conda /bin/2.83/python
#     # && echo $WORKING_DIR/scene_generation >> /bin/2.83/python/lib/python3.7/site-packages/scene_generation.pth

RUN wget https://download.blender.org/release/Blender2.83/blender-2.83.20-linux-x64.tar.xz \
	&& tar -xvf blender-2.83.20* --strip-components=1 -C /bin \
	&& rm -rf blender-2.83.20* \
	&& rm -rf blender-2.83.20* \
    && rm -rf /bin/2.83/python \
    && ln -s /opt/conda /bin/2.83/python
    # && echo $WORKING_DIR/scene_generation >> /bin/2.83/python/lib/python3.7/site-packages/scene_generation.pth

# RUN mkdir -p /usr/local/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /usr/local/.mujoco \
#     && rm mujoco.tar.gz

# ENV LD_LIBRARY_PATH /usr/local/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
# ENV MUJOCO_PY_MUJOCO_PATH /usr/local/.mujoco/mujoco210

# RUN chown 777 -R /usr/local/.mujoco
# RUN chmod 777 -R /usr/local/.mujoco

RUN pip install opencv-python pycocotools
RUN pip install -U mujoco-py==2.1.2.14
RUN pip install pip install robosuite
RUN chown 777 -R /opt/conda/lib/python3.7/site-packages/mujoco_py
RUN chmod 777 -R /opt/conda/lib/python3.7/site-packages/mujoco_py
RUN rm /opt/conda/lib/python3.7/site-packages/robosuite/models/grippers/robotiq_85_gripper.py
RUN rm /opt/conda/lib/python3.7/site-packages/robosuite/models/grippers/__init__.py
RUN rm /opt/conda/lib/python3.7/site-packages/robosuite/models/assets/grippers/robotiq_gripper_85.xml
COPY robotiq_85_gripper.py /opt/conda/lib/python3.7/site-packages/robosuite/models/grippers
COPY __init__.py /opt/conda/lib/python3.7/site-packages/robosuite/models/grippers
COPY robotiq_gripper_85.xml /opt/conda/lib/python3.7/site-packages/robosuite/models/assets/grippers
RUN chown -R 777 /opt/conda/lib/python3.7/site-packages/robosuite
RUN chmod -R 777 /opt/conda/lib/python3.7/site-packages/robosuite
RUN python -c "import mujoco_py"
