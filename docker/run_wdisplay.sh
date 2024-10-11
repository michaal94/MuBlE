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
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --env="DISPLAY" \
    --net=host \
    -w $WORKING_DIR \
    --name ns_ap_container \
    ns_ap