## Build the container for the llmclients

```
docker build -f Dockerfile.llmclients -t llm-clients:1.0.0 -t llm-clients:latest .

docker run -d -it  --name llm-clients -h llm-clients -v `pwd`:/app/llm-clients/src --shm-size=8g -p 8881:8888  --ulimit memlock=-1 --ulimit stack=67108864 llm-clients:latest
```