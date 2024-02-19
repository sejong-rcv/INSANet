IMAGE_NAME = INSANet
IMAGE_TAG = maintainer
PYTHON_VERSION = 3.7.16
CONDA_ENV_NAME = insanet
UID := $(shell id -u)
USER_NAME := $(shell whoami)
POS_GPUS=all

docker-make:
	$(eval IMAGE_NAME := $(IMAGE_NAME):$(IMAGE_TAG))
	@nvidia-docker build -t ${IMAGE_NAME} \
  	--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
  	--build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
  	-f ./Dockerfile \
  	.

# Set the volume
docker-run:
	@nvidia-docker run -it -u ${USER_NAME} \
		-p ${PORT}:${PORT} \
		--name ${CONTAINER_NAME} \
		-w /home/${USER_NAME}/workspace/ \
		-e "TERM=xterm-256color" \
		-e NVIDIA_VISIBLE_DEVICES=${POS_GPUS} \
		-v /home/${USER_NAME}/workspace:/home/${USER_NAME}/workspace \
		--shm-size=32G \
		--ipc=host \
		${USER_NAME}/${IMAGE_NAME}:${IMAGE_TAG} \
		/bin/bash 