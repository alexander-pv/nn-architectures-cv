
### nn-architectures

Notes about neural network architectures in 2D computer vision

How to run:

Conda:


```bash
$ git clone nn-architectures-cv && cd nn-architectures-cv && git lfs pull
$ conda env create -f environment.yml && conda activate nn-architectures
$ jupyter-notebook
```

Docker:

Prerequisites:
 - [docker engine](https://docs.docker.com/engine/install/)
 - [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# DockerHub image
$ docker pull alrdockerhub/nn-architectures-cv:latest
$ docker run --gpus=all --runtime nvidia \
  --net=host --rm -it \
  --name nn-architectures-cv alrdockerhub/nn-architectures-cv:latest
  
# Local build
$ docker build -t nn-architectures-cv:latest .
$ docker run --gpus=all --runtime nvidia \
  --net=host --rm -it \
  --name nn-architectures-cv nn-architectures-cv:latest
```