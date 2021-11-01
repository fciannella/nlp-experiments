# nlp-experiments

## Docker commands to build and run

```
docker build -t nlp-experiments:1.0.0 -t nlp-experiments:latest . 

docker run --gpus all -d -it --rm -v /mnt/nvdl/usr/fciannella/src/nlp-experiments:/app/src --shm-size=8g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nlp-experiments:latest
```