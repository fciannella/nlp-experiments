## Build the container

```
docker build -f Dockerfile.GPTJ -t fciannella/gptj:1.0 -t fciannella/gptj:latest .

docker run -d -ti --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network=host --ipc=host -it -v /mnt/nvdl/usr/fciannella/src/gptj-data/:/gptj-data --name=gptj-container fciannella/gptj:latest

```