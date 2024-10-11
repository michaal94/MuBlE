WORKING_DIR=$(realpath $(pwd)/..)
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t ns_ap --build-arg WORKING_DIR=$WORKING_DIR .


# DOCKER_BUILDKIT=1 \
#     docker build \
#         --progress=auto \
#         --target dev \
#         -t test \
#         --build-arg BASE_IMAGE=ubuntu:20.04 \
# 							--build-arg PYTHON_VERSION=3.9.7 \
# 							--build-arg CUDA_VERSION=11.3 \
# 							--build-arg CUDA_CHANNEL=nvidia \
# 							--build-arg PYTORCH_VERSION=1.11.0 \
# 							--build-arg INSTALL_CHANNEL=pytorch .