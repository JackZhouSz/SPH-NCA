SPH-NCA
=============

[[Interactive Demo](https://hyunsoo0o0o0.github.io/SPH-NCA/)]

![Demo page example](docs/media/demo-page.gif)

Official repository for SIGGRAPH 2025 Poster `Train Once, Generate Anywhere: Discretization Agnostic Neural Cellular Automata using SPH Method`


## Setting up

- NVIDIA GPU and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is *required*. Tested environment uses CUDA 12.2, but any CUDA 11.8+ should work.
- Python 3.12 is used

### Run in a container
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) is *required*.
```bash
$ docker compose up --build -d 
# May take minutes for the first time
$ docker compose exec sph-nca bash
# Now in a docker container
$ nvidia-smi # Is GPU available in the container?
```

### Run in a virtual environment

#### Using `pip`
```bash
# Set up the virtual env
$ python3 -m venv venv
$ source venv/bin/activate

# Install requirements.txt
$ pip install -r requirements.txt
```

#### Using `conda`/`mamba`
Use `environment.yml`, which is the same package configuration as used to configure the container. I don't have conda/mamba in my workspace, so it's not tested outside the container.

## Run the code

### Training
```bash
$ cd code
$ bash train-example.sh # Try different examples in train-example.sh
# Weight checkpoint will be generated in code/checkpoints/
```


### Testing
```bash
$ bash test-example.sh ./checkpoints/GENERATED_CHECKPOINT.pt
# Images/point clouds will be generated in code/output/
```


## Project structure

- `code/`: Contains the core Python implementation for SPH-NCA. The paper mainly uses this code.
    - `commons/`: Utility functions and common modules used across the project, including geometry, indexing, and sampling.
    - `data/`: Example data files used for training and testing, which have different licenses so be aware!
        - [License for Stanford bunny](https://graphics.stanford.edu/data/3Dscanrep/)
        - [License for Describable Texture Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)
    - `sphops/`: Differentiable SPH operations with Numba CUDA.
- `docs/`: A web-based demonstration of the SPH-NCA model. Almost every line is vibe-coded so don't expect much :/
    - `weights/`: Pre-trained weights.
- `requirements.txt`: Python package dependencies for `pip` and `venv`. Tested environment (i.e., mine) use this.
- `environment.yml`: Same, but for Conda/Mamba. Docker uses this.
- `Dockerfile`: For quick environment setup with Docker
- `docker-compose.yml`: For quicker setup with Docker compose who always forgets Docker commands (like me)
