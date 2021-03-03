# nvidiapy 

A (user-)friendly wrapper to `nvidia-smi`.

It can be used to filter the GPUs based on resource usage (e.g. to choose the least utilized GPU on a multi-GPU system).

## Usage

### CLI

```
nvsmi --help
nvsmi ls --help
nvsmi ps --help
```

### As a library

```
import nvsmi

nvidiapy.get_gpus()
nvidiapy.get_available_gpus()
nvidiapy.get_gpu_processes()
```

## Prerequisites

- An nvidia GPU
- `nvidia-smi`
- Python 2.7 or 3.6+

## Installation

### pip

```
pip install --user nvsmi
```
