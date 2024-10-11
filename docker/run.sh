# export UID=$(id -u)
# export GID=$(id -g)
WORKING_DIR=$(realpath $(pwd)/..)

docker run -it \
    --rm \
    --gpus 'all' \
    --user $(id -u):$(id -g) \
    -v $WORKING_DIR:$WORKING_DIR \
    -v="/etc/group:/etc/group:ro" \
    -v="/etc/passwd:/etc/passwd:ro" \
    -v="/etc/shadow:/etc/shadow:ro" \
    -w $WORKING_DIR \
    --name ns_ap_container \
    --shm-size=16gb \
    --privileged \
    --ulimit memlock=65536:65536 \
    ns_ap