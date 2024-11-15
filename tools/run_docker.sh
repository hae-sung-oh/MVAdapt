sudo docker run -it --mount source=/path/to/MVAdapt/results,target=/workspace/results,type=bind --rm --net=host --gpus '"device=0"' -e PORT=2000 transfuser-agent:latest

