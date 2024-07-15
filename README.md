# TensorRT-LLM - Llama 3 1M Context

##### https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#1m-long-context-test-case

### 15 July 2024
### nvcr.io/nvidia/rapidsai/notebooks:24.04-cuda12.0-py3.10

  

### VM Specs


```python
!uname -a
```

    Linux verb-workspace 6.2.0-37-generic #38~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov  2 18:01:13 UTC 2 x86_64 x86_64 x86_64 GNU/Linux



```python
!cat /etc/lsb-release
```

    DISTRIB_ID=Ubuntu
    DISTRIB_RELEASE=22.04
    DISTRIB_CODENAME=jammy
    DISTRIB_DESCRIPTION="Ubuntu 22.04.4 LTS"



```python
!nvidia-smi
```

    Mon Jul 15 22:30:14 2024       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
    | N/A   31C    P0              42W / 400W |      7MiB / 40960MiB |      0%      Default |
    |                                         |                      |             Disabled |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+



```python
!free -h
```

                   total        used        free      shared  buff/cache   available
    Mem:           216Gi       2.0Gi        25Gi       5.0Mi       188Gi       212Gi
    Swap:             0B          0B          0B



```python
!nproc
```

    30



```python
!python -V
```

    Python 3.10.14


   

# Install System & Python Dependencies   
https://nvidia.github.io/TensorRT-LLM/installation/linux.html


```python
%%time 

!apt-get -y install python3.10 python3-pip python3.10-dev  openmpi-bin libopenmpi-dev git git-lfs python3-mpi4py
```

    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    libopenmpi-dev is already the newest version (4.1.2-2ubuntu1).
    openmpi-bin is already the newest version (4.1.2-2ubuntu1).
    python3-mpi4py is already the newest version (3.1.3-1build2).
    git is already the newest version (1:2.34.1-1ubuntu1.11).
    python3.10 is already the newest version (3.10.12-1~22.04.4).
    python3.10-dev is already the newest version (3.10.12-1~22.04.4).
    git-lfs is already the newest version (3.0.2-1ubuntu0.2).
    python3-pip is already the newest version (22.0.2+dfsg-1ubuntu0.4).
    0 upgraded, 0 newly installed, 0 to remove and 23 not upgraded.
    CPU times: user 21.9 ms, sys: 11.3 ms, total: 33.2 ms
    Wall time: 1.28 s


  

  

# Install TensorRT-LLM
### For the latest verions, use the '--pre' command line option
##### This is needed for the 1M Context use case provided in this notebook


```python
%%time
# 4+ minutes

# latest version —> add --pre after ‘-U’
# stable version —> no `--pre` option. Will be v0.10 (15 Jul '24)

!pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
```

    Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com
    Requirement already satisfied: tensorrt_llm in /opt/conda/lib/python3.10/site-packages (0.12.0.dev2024070900)
    Requirement already satisfied: accelerate>=0.25.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.32.1)
    Requirement already satisfied: build in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (1.2.1)
    Requirement already satisfied: colored in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (2.2.4)
    Requirement already satisfied: cuda-python in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (12.5.0)
    Requirement already satisfied: diffusers>=0.27.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.29.2)
    Requirement already satisfied: lark in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (1.1.9)
    Requirement already satisfied: mpi4py in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (3.1.6)
    Requirement already satisfied: numpy<2 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (1.26.4)
    Requirement already satisfied: onnx>=1.12.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (1.16.1)
    Requirement already satisfied: polygraphy in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.49.9)
    Requirement already satisfied: psutil in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (6.0.0)
    Requirement already satisfied: pynvml>=11.5.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (11.5.2)
    Requirement already satisfied: pulp in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (2.9.0)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (2.2.2)
    Requirement already satisfied: h5py==3.10.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (3.10.0)
    Requirement already satisfied: StrEnum in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.4.15)
    Requirement already satisfied: sentencepiece>=0.1.99 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.2.0)
    Requirement already satisfied: torch<=2.4.0a0,>=2.3.0a0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (2.3.1)
    Requirement already satisfied: nvidia-modelopt<0.14,~=0.13 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.13.1)
    Requirement already satisfied: transformers>=4.38.2 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (4.42.4)
    Requirement already satisfied: pillow==10.3.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (10.3.0)
    Requirement already satisfied: wheel in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.43.0)
    Requirement already satisfied: optimum in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (1.21.2)
    Requirement already satisfied: evaluate in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (0.4.2)
    Requirement already satisfied: janus in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (1.0.0)
    Requirement already satisfied: mpmath>=1.3.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (1.3.0)
    Requirement already satisfied: tensorrt-cu12==10.1.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt_llm) (10.1.0)
    Requirement already satisfied: tensorrt-cu12-libs==10.1.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt-cu12==10.1.0->tensorrt_llm) (10.1.0)
    Requirement already satisfied: tensorrt-cu12-bindings==10.1.0 in /opt/conda/lib/python3.10/site-packages (from tensorrt-cu12==10.1.0->tensorrt_llm) (10.1.0)
    Requirement already satisfied: nvidia-cuda-runtime-cu12 in /opt/conda/lib/python3.10/site-packages (from tensorrt-cu12-libs==10.1.0->tensorrt-cu12==10.1.0->tensorrt_llm) (12.1.105)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from accelerate>=0.25.0->tensorrt_llm) (24.1)
    Requirement already satisfied: pyyaml in /opt/conda/lib/python3.10/site-packages (from accelerate>=0.25.0->tensorrt_llm) (6.0.1)
    Requirement already satisfied: huggingface-hub in /opt/conda/lib/python3.10/site-packages (from accelerate>=0.25.0->tensorrt_llm) (0.23.4)
    Requirement already satisfied: safetensors>=0.3.1 in /opt/conda/lib/python3.10/site-packages (from accelerate>=0.25.0->tensorrt_llm) (0.4.3)
    Requirement already satisfied: importlib-metadata in /opt/conda/lib/python3.10/site-packages (from diffusers>=0.27.0->tensorrt_llm) (8.0.0)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from diffusers>=0.27.0->tensorrt_llm) (3.15.4)
    Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.10/site-packages (from diffusers>=0.27.0->tensorrt_llm) (2024.5.15)
    Requirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from diffusers>=0.27.0->tensorrt_llm) (2.32.3)
    Requirement already satisfied: cloudpickle>=1.6.0 in /opt/conda/lib/python3.10/site-packages (from nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (3.0.0)
    Requirement already satisfied: ninja in /opt/conda/lib/python3.10/site-packages (from nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (1.11.1.1)
    Requirement already satisfied: pydantic>=2.0 in /opt/conda/lib/python3.10/site-packages (from nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (2.8.2)
    Requirement already satisfied: rich in /opt/conda/lib/python3.10/site-packages (from nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (13.7.1)
    Requirement already satisfied: scipy in /opt/conda/lib/python3.10/site-packages (from nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (1.14.0)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.10/site-packages (from nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (4.66.4)
    Requirement already satisfied: protobuf>=3.20.2 in /opt/conda/lib/python3.10/site-packages (from onnx>=1.12.0->tensorrt_llm) (5.27.2)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (4.8.0)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (1.13.0)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (3.3)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (3.1.4)
    Requirement already satisfied: fsspec in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (2024.3.1)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (12.1.105)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (12.1.105)
    Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (8.9.2.26)
    Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (12.1.3.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (11.0.2.54)
    Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (10.3.2.106)
    Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (11.4.5.107)
    Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (12.1.0.106)
    Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (2.20.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (12.1.105)
    Requirement already satisfied: triton==2.3.1 in /opt/conda/lib/python3.10/site-packages (from torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (2.3.1)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in /opt/conda/lib/python3.10/site-packages (from nvidia-cusolver-cu12==11.4.5.107->torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (12.5.82)
    Requirement already satisfied: tokenizers<0.20,>=0.19 in /opt/conda/lib/python3.10/site-packages (from transformers>=4.38.2->tensorrt_llm) (0.19.1)
    Requirement already satisfied: pyproject_hooks in /opt/conda/lib/python3.10/site-packages (from build->tensorrt_llm) (1.1.0)
    Requirement already satisfied: tomli>=1.1.0 in /opt/conda/lib/python3.10/site-packages (from build->tensorrt_llm) (2.0.1)
    Requirement already satisfied: datasets>=2.0.0 in /opt/conda/lib/python3.10/site-packages (from evaluate->tensorrt_llm) (2.19.2)
    Requirement already satisfied: dill in /opt/conda/lib/python3.10/site-packages (from evaluate->tensorrt_llm) (0.3.8)
    Requirement already satisfied: xxhash in /opt/conda/lib/python3.10/site-packages (from evaluate->tensorrt_llm) (3.4.1)
    Requirement already satisfied: multiprocess in /opt/conda/lib/python3.10/site-packages (from evaluate->tensorrt_llm) (0.70.16)
    Requirement already satisfied: coloredlogs in /opt/conda/lib/python3.10/site-packages (from optimum->tensorrt_llm) (15.0.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.10/site-packages (from pandas->tensorrt_llm) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas->tensorrt_llm) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.10/site-packages (from pandas->tensorrt_llm) (2024.1)
    Requirement already satisfied: pyarrow>=12.0.0 in /opt/conda/lib/python3.10/site-packages (from datasets>=2.0.0->evaluate->tensorrt_llm) (16.1.0)
    Requirement already satisfied: pyarrow-hotfix in /opt/conda/lib/python3.10/site-packages (from datasets>=2.0.0->evaluate->tensorrt_llm) (0.6)
    Requirement already satisfied: aiohttp in /opt/conda/lib/python3.10/site-packages (from datasets>=2.0.0->evaluate->tensorrt_llm) (3.9.5)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from pydantic>=2.0->nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /opt/conda/lib/python3.10/site-packages (from pydantic>=2.0->nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (2.20.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas->tensorrt_llm) (1.16.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->diffusers>=0.27.0->tensorrt_llm) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->diffusers>=0.27.0->tensorrt_llm) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests->diffusers>=0.27.0->tensorrt_llm) (2.2.2)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests->diffusers>=0.27.0->tensorrt_llm) (2024.7.4)
    Requirement already satisfied: humanfriendly>=9.1 in /opt/conda/lib/python3.10/site-packages (from coloredlogs->optimum->tensorrt_llm) (10.0)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/lib/python3.10/site-packages (from importlib-metadata->diffusers>=0.27.0->tensorrt_llm) (3.19.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch<=2.4.0a0,>=2.3.0a0->tensorrt_llm) (2.1.5)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/conda/lib/python3.10/site-packages (from rich->nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/conda/lib/python3.10/site-packages (from rich->nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (2.18.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.0.0->evaluate->tensorrt_llm) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.0.0->evaluate->tensorrt_llm) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.0.0->evaluate->tensorrt_llm) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.0.0->evaluate->tensorrt_llm) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.0.0->evaluate->tensorrt_llm) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.0.0->evaluate->tensorrt_llm) (4.0.3)
    Requirement already satisfied: mdurl~=0.1 in /opt/conda/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->nvidia-modelopt<0.14,~=0.13->tensorrt_llm) (0.1.2)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCPU times: user 58.9 ms, sys: 27.1 ms, total: 86 ms
    Wall time: 5.23 s


  

  

### Test TensorRT-LLM Installation


```python
import tensorrt_llm
```

    [TensorRT-LLM] TensorRT-LLM version: 0.12.0.dev2024070900



```python
tensorrt_llm.__version__
```




    '0.12.0.dev2024070900'



  

  

# git clone TensorRT-LLM code
### 3 minutes


```python
%%time 
# 13 seconds

!git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

    Cloning into 'TensorRT-LLM'...
    remote: Enumerating objects: 19939, done.[K
    remote: Counting objects: 100% (9518/9518), done.[K
    remote: Compressing objects: 100% (2362/2362), done.[K
    remote: Total 19939 (delta 7631), reused 8358 (delta 7118), pack-reused 10421[K
    Receiving objects: 100% (19939/19939), 298.50 MiB | 56.48 MiB/s, done.
    Resolving deltas: 100% (14668/14668), done.
    Updating files: 100% (2422/2422), done.
    Filtering content: 100% (14/14), 212.51 MiB | 113.75 MiB/s, done.
    CPU times: user 151 ms, sys: 66.8 ms, total: 218 ms
    Wall time: 13.5 s


### Intall dependencies


```python
!cat TensorRT-LLM/requirements-dev.txt
```

    -r requirements.txt
    datasets==2.19.2
    einops
    graphviz
    mypy
    parameterized
    pre-commit
    pybind11
    pybind11-stubgen
    pytest-cov
    pytest-forked
    pytest-xdist
    rouge_score
    cloudpickle
    typing-extensions==4.8.0
    bandit==1.7.7
    jsonlines==4.0.0
    jieba==0.42.1
    rouge==1.0.1



```python
%%time 
# 2.5 minutes
!pip install -r TensorRT-LLM/requirements-dev.txt
```

    Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com
    Ignoring tensorrt: markers 'platform_machine == "aarch64"' don't match your environment
    Collecting accelerate>=0.25.0 (from -r TensorRT-LLM/requirements.txt (line 2))
      Using cached accelerate-0.32.1-py3-none-any.whl.metadata (18 kB)
    Collecting build (from -r TensorRT-LLM/requirements.txt (line 3))
      Using cached build-1.2.1-py3-none-any.whl.metadata (4.3 kB)
    Collecting colored (from -r TensorRT-LLM/requirements.txt (line 4))
      Using cached colored-2.2.4-py3-none-any.whl.metadata (3.6 kB)
    Collecting cuda-python (from -r TensorRT-LLM/requirements.txt (line 5))
      Using cached cuda_python-12.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
    Collecting diffusers>=0.27.0 (from -r TensorRT-LLM/requirements.txt (line 6))
      Using cached diffusers-0.29.2-py3-none-any.whl.metadata (19 kB)
    Collecting lark (from -r TensorRT-LLM/requirements.txt (line 7))
      Using cached lark-1.1.9-py3-none-any.whl.metadata (1.9 kB)
    Collecting mpi4py (from -r TensorRT-LLM/requirements.txt (line 8))
      Using cached mpi4py-3.1.6-cp310-cp310-linux_x86_64.whl
    Collecting numpy<2 (from -r TensorRT-LLM/requirements.txt (line 9))
      Using cached numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
    Collecting onnx>=1.12.0 (from -r TensorRT-LLM/requirements.txt (line 10))
      Using cached onnx-1.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
    Collecting polygraphy (from -r TensorRT-LLM/requirements.txt (line 11))
      Using cached https://pypi.nvidia.com/polygraphy/polygraphy-0.49.9-py2.py3-none-any.whl (346 kB)
    Collecting psutil (from -r TensorRT-LLM/requirements.txt (line 12))
      Using cached psutil-6.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)
    Collecting pynvml>=11.5.0 (from -r TensorRT-LLM/requirements.txt (line 13))
      Using cached pynvml-11.5.2-py3-none-any.whl.metadata (8.8 kB)
    Collecting pulp (from -r TensorRT-LLM/requirements.txt (line 14))
      Using cached PuLP-2.9.0-py3-none-any.whl.metadata (5.4 kB)
    Collecting pandas (from -r TensorRT-LLM/requirements.txt (line 15))
      Using cached pandas-2.2.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (19 kB)
    Collecting h5py==3.10.0 (from -r TensorRT-LLM/requirements.txt (line 16))
      Using cached h5py-3.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)
    Collecting StrEnum (from -r TensorRT-LLM/requirements.txt (line 17))
      Using cached StrEnum-0.4.15-py3-none-any.whl.metadata (5.3 kB)
    Collecting sentencepiece>=0.1.99 (from -r TensorRT-LLM/requirements.txt (line 18))
      Using cached sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
    Collecting tensorrt-cu12==10.1.0 (from -r TensorRT-LLM/requirements.txt (line 19))
      Using cached tensorrt_cu12-10.1.0-py2.py3-none-any.whl
    Collecting torch<=2.4.0a0,>=2.3.0a0 (from -r TensorRT-LLM/requirements.txt (line 23))
      Using cached torch-2.3.1-cp310-cp310-manylinux1_x86_64.whl.metadata (26 kB)
    Collecting nvidia-modelopt<0.14,~=0.13 (from -r TensorRT-LLM/requirements.txt (line 24))
      Using cached https://pypi.nvidia.com/nvidia-modelopt/nvidia_modelopt-0.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
    Collecting transformers>=4.38.2 (from -r TensorRT-LLM/requirements.txt (line 25))
      Using cached transformers-4.42.4-py3-none-any.whl.metadata (43 kB)
    Collecting pillow==10.3.0 (from -r TensorRT-LLM/requirements.txt (line 26))
      Using cached pillow-10.3.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (9.2 kB)
    Collecting wheel (from -r TensorRT-LLM/requirements.txt (line 27))
      Using cached wheel-0.43.0-py3-none-any.whl.metadata (2.2 kB)
    Collecting optimum (from -r TensorRT-LLM/requirements.txt (line 28))
      Using cached optimum-1.21.2-py3-none-any.whl.metadata (19 kB)
    Collecting evaluate (from -r TensorRT-LLM/requirements.txt (line 29))
      Using cached evaluate-0.4.2-py3-none-any.whl.metadata (9.3 kB)
    Collecting janus (from -r TensorRT-LLM/requirements.txt (line 30))
      Using cached janus-1.0.0-py3-none-any.whl.metadata (4.5 kB)
    Collecting mpmath>=1.3.0 (from -r TensorRT-LLM/requirements.txt (line 31))
      Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
    Collecting datasets==2.19.2 (from -r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached datasets-2.19.2-py3-none-any.whl.metadata (19 kB)
    Collecting einops (from -r TensorRT-LLM/requirements-dev.txt (line 3))
      Using cached einops-0.8.0-py3-none-any.whl.metadata (12 kB)
    Collecting graphviz (from -r TensorRT-LLM/requirements-dev.txt (line 4))
      Using cached graphviz-0.20.3-py3-none-any.whl.metadata (12 kB)
    Collecting mypy (from -r TensorRT-LLM/requirements-dev.txt (line 5))
      Using cached mypy-1.10.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.9 kB)
    Collecting parameterized (from -r TensorRT-LLM/requirements-dev.txt (line 6))
      Using cached parameterized-0.9.0-py2.py3-none-any.whl.metadata (18 kB)
    Collecting pre-commit (from -r TensorRT-LLM/requirements-dev.txt (line 7))
      Using cached pre_commit-3.7.1-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting pybind11 (from -r TensorRT-LLM/requirements-dev.txt (line 8))
      Using cached pybind11-2.13.1-py3-none-any.whl.metadata (9.5 kB)
    Collecting pybind11-stubgen (from -r TensorRT-LLM/requirements-dev.txt (line 9))
      Using cached pybind11_stubgen-2.5.1-py3-none-any.whl.metadata (1.7 kB)
    Collecting pytest-cov (from -r TensorRT-LLM/requirements-dev.txt (line 10))
      Using cached pytest_cov-5.0.0-py3-none-any.whl.metadata (27 kB)
    Collecting pytest-forked (from -r TensorRT-LLM/requirements-dev.txt (line 11))
      Using cached pytest_forked-1.6.0-py3-none-any.whl.metadata (3.5 kB)
    Collecting pytest-xdist (from -r TensorRT-LLM/requirements-dev.txt (line 12))
      Using cached pytest_xdist-3.6.1-py3-none-any.whl.metadata (4.3 kB)
    Collecting rouge_score (from -r TensorRT-LLM/requirements-dev.txt (line 13))
      Using cached rouge_score-0.1.2-py3-none-any.whl
    Collecting cloudpickle (from -r TensorRT-LLM/requirements-dev.txt (line 14))
      Using cached cloudpickle-3.0.0-py3-none-any.whl.metadata (7.0 kB)
    Collecting typing-extensions==4.8.0 (from -r TensorRT-LLM/requirements-dev.txt (line 15))
      Using cached typing_extensions-4.8.0-py3-none-any.whl.metadata (3.0 kB)
    Collecting bandit==1.7.7 (from -r TensorRT-LLM/requirements-dev.txt (line 16))
      Using cached bandit-1.7.7-py3-none-any.whl.metadata (5.9 kB)
    Collecting jsonlines==4.0.0 (from -r TensorRT-LLM/requirements-dev.txt (line 17))
      Using cached jsonlines-4.0.0-py3-none-any.whl.metadata (1.6 kB)
    Collecting jieba==0.42.1 (from -r TensorRT-LLM/requirements-dev.txt (line 18))
      Using cached jieba-0.42.1-py3-none-any.whl
    Collecting rouge==1.0.1 (from -r TensorRT-LLM/requirements-dev.txt (line 19))
      Using cached rouge-1.0.1-py3-none-any.whl.metadata (4.1 kB)
    Collecting tensorrt-cu12-libs==10.1.0 (from tensorrt-cu12==10.1.0->-r TensorRT-LLM/requirements.txt (line 19))
      Using cached https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.1.0-py2.py3-none-manylinux_2_17_x86_64.whl (1056.3 MB)
    Collecting tensorrt-cu12-bindings==10.1.0 (from tensorrt-cu12==10.1.0->-r TensorRT-LLM/requirements.txt (line 19))
      Using cached https://pypi.nvidia.com/tensorrt-cu12-bindings/tensorrt_cu12_bindings-10.1.0-cp310-none-manylinux_2_17_x86_64.whl (1.1 MB)
    Collecting filelock (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached filelock-3.15.4-py3-none-any.whl.metadata (2.9 kB)
    Collecting pyarrow>=12.0.0 (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached pyarrow-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (3.0 kB)
    Collecting pyarrow-hotfix (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached pyarrow_hotfix-0.6-py3-none-any.whl.metadata (3.6 kB)
    Collecting dill<0.3.9,>=0.3.0 (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached dill-0.3.8-py3-none-any.whl.metadata (10 kB)
    Collecting requests>=2.32.1 (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
    Collecting tqdm>=4.62.1 (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached tqdm-4.66.4-py3-none-any.whl.metadata (57 kB)
    Collecting xxhash (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached xxhash-3.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
    Collecting multiprocess (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached multiprocess-0.70.16-py310-none-any.whl.metadata (7.2 kB)
    Collecting fsspec<=2024.3.1,>=2023.1.0 (from fsspec[http]<=2024.3.1,>=2023.1.0->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached fsspec-2024.3.1-py3-none-any.whl.metadata (6.8 kB)
    Collecting aiohttp (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached aiohttp-3.9.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.5 kB)
    Collecting huggingface-hub>=0.21.2 (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached huggingface_hub-0.23.4-py3-none-any.whl.metadata (12 kB)
    Collecting packaging (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached packaging-24.1-py3-none-any.whl.metadata (3.2 kB)
    Collecting pyyaml>=5.1 (from datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
    Collecting stevedore>=1.20.0 (from bandit==1.7.7->-r TensorRT-LLM/requirements-dev.txt (line 16))
      Using cached stevedore-5.2.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting rich (from bandit==1.7.7->-r TensorRT-LLM/requirements-dev.txt (line 16))
      Using cached rich-13.7.1-py3-none-any.whl.metadata (18 kB)
    Collecting attrs>=19.2.0 (from jsonlines==4.0.0->-r TensorRT-LLM/requirements-dev.txt (line 17))
      Using cached attrs-23.2.0-py3-none-any.whl.metadata (9.5 kB)
    Collecting six (from rouge==1.0.1->-r TensorRT-LLM/requirements-dev.txt (line 19))
      Using cached six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
    Collecting nvidia-cuda-runtime-cu12 (from tensorrt-cu12-libs==10.1.0->tensorrt-cu12==10.1.0->-r TensorRT-LLM/requirements.txt (line 19))
      Using cached https://pypi.nvidia.com/nvidia-cuda-runtime-cu12/nvidia_cuda_runtime_cu12-12.5.82-py3-none-manylinux2014_x86_64.whl (895 kB)
    Collecting safetensors>=0.3.1 (from accelerate>=0.25.0->-r TensorRT-LLM/requirements.txt (line 2))
      Using cached safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
    Collecting pyproject_hooks (from build->-r TensorRT-LLM/requirements.txt (line 3))
      Using cached pyproject_hooks-1.1.0-py3-none-any.whl.metadata (1.3 kB)
    Collecting tomli>=1.1.0 (from build->-r TensorRT-LLM/requirements.txt (line 3))
      Using cached tomli-2.0.1-py3-none-any.whl.metadata (8.9 kB)
    Collecting importlib-metadata (from diffusers>=0.27.0->-r TensorRT-LLM/requirements.txt (line 6))
      Using cached importlib_metadata-8.0.0-py3-none-any.whl.metadata (4.6 kB)
    Collecting regex!=2019.12.17 (from diffusers>=0.27.0->-r TensorRT-LLM/requirements.txt (line 6))
      Using cached regex-2024.5.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)
    Collecting protobuf>=3.20.2 (from onnx>=1.12.0->-r TensorRT-LLM/requirements.txt (line 10))
      Using cached protobuf-5.27.2-cp38-abi3-manylinux2014_x86_64.whl.metadata (592 bytes)
    Collecting python-dateutil>=2.8.2 (from pandas->-r TensorRT-LLM/requirements.txt (line 15))
      Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
    Collecting pytz>=2020.1 (from pandas->-r TensorRT-LLM/requirements.txt (line 15))
      Using cached pytz-2024.1-py2.py3-none-any.whl.metadata (22 kB)
    Collecting tzdata>=2022.7 (from pandas->-r TensorRT-LLM/requirements.txt (line 15))
      Using cached tzdata-2024.1-py2.py3-none-any.whl.metadata (1.4 kB)
    Collecting sympy (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached sympy-1.13.0-py3-none-any.whl.metadata (12 kB)
    Collecting networkx (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached networkx-3.3-py3-none-any.whl.metadata (5.1 kB)
    Collecting jinja2 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached jinja2-3.1.4-py3-none-any.whl.metadata (2.6 kB)
    Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-cuda-nvrtc-cu12/nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    Collecting nvidia-cuda-runtime-cu12 (from tensorrt-cu12-libs==10.1.0->tensorrt-cu12==10.1.0->-r TensorRT-LLM/requirements.txt (line 19))
      Using cached https://pypi.nvidia.com/nvidia-cuda-runtime-cu12/nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-cuda-cupti-cu12/nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-cudnn-cu12/nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    Collecting nvidia-cublas-cu12==12.1.3.1 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-cublas-cu12/nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    Collecting nvidia-cufft-cu12==11.0.2.54 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-cufft-cu12/nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    Collecting nvidia-curand-cu12==10.3.2.106 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-curand-cu12/nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-cusolver-cu12/nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-cusparse-cu12/nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    Collecting nvidia-nccl-cu12==2.20.5 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-nccl-cu12/nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl (176.2 MB)
    Collecting nvidia-nvtx-cu12==12.1.105 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-nvtx-cu12/nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    Collecting triton==2.3.1 (from torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached triton-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.4 kB)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached https://pypi.nvidia.com/nvidia-nvjitlink-cu12/nvidia_nvjitlink_cu12-12.5.82-py3-none-manylinux2014_x86_64.whl (21.3 MB)
    Collecting ninja (from nvidia-modelopt<0.14,~=0.13->-r TensorRT-LLM/requirements.txt (line 24))
      Using cached ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl.metadata (5.3 kB)
    Collecting pydantic>=2.0 (from nvidia-modelopt<0.14,~=0.13->-r TensorRT-LLM/requirements.txt (line 24))
      Using cached pydantic-2.8.2-py3-none-any.whl.metadata (125 kB)
    Collecting scipy (from nvidia-modelopt<0.14,~=0.13->-r TensorRT-LLM/requirements.txt (line 24))
      Using cached scipy-1.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
    Collecting tokenizers<0.20,>=0.19 (from transformers>=4.38.2->-r TensorRT-LLM/requirements.txt (line 25))
      Using cached tokenizers-0.19.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
    Collecting coloredlogs (from optimum->-r TensorRT-LLM/requirements.txt (line 28))
      Using cached coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
    Collecting mypy-extensions>=1.0.0 (from mypy->-r TensorRT-LLM/requirements-dev.txt (line 5))
      Using cached mypy_extensions-1.0.0-py3-none-any.whl.metadata (1.1 kB)
    Collecting cfgv>=2.0.0 (from pre-commit->-r TensorRT-LLM/requirements-dev.txt (line 7))
      Using cached cfgv-3.4.0-py2.py3-none-any.whl.metadata (8.5 kB)
    Collecting identify>=1.0.0 (from pre-commit->-r TensorRT-LLM/requirements-dev.txt (line 7))
      Using cached identify-2.6.0-py2.py3-none-any.whl.metadata (4.4 kB)
    Collecting nodeenv>=0.11.1 (from pre-commit->-r TensorRT-LLM/requirements-dev.txt (line 7))
      Using cached nodeenv-1.9.1-py2.py3-none-any.whl.metadata (21 kB)
    Collecting virtualenv>=20.10.0 (from pre-commit->-r TensorRT-LLM/requirements-dev.txt (line 7))
      Using cached virtualenv-20.26.3-py3-none-any.whl.metadata (4.5 kB)
    Collecting pytest>=4.6 (from pytest-cov->-r TensorRT-LLM/requirements-dev.txt (line 10))
      Using cached pytest-8.2.2-py3-none-any.whl.metadata (7.6 kB)
    Collecting coverage>=5.2.1 (from coverage[toml]>=5.2.1->pytest-cov->-r TensorRT-LLM/requirements-dev.txt (line 10))
      Using cached coverage-7.6.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.2 kB)
    Collecting py (from pytest-forked->-r TensorRT-LLM/requirements-dev.txt (line 11))
      Using cached py-1.11.0-py2.py3-none-any.whl.metadata (2.8 kB)
    Collecting execnet>=2.1 (from pytest-xdist->-r TensorRT-LLM/requirements-dev.txt (line 12))
      Using cached execnet-2.1.1-py3-none-any.whl.metadata (2.9 kB)
    Collecting absl-py (from rouge_score->-r TensorRT-LLM/requirements-dev.txt (line 13))
      Using cached absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting nltk (from rouge_score->-r TensorRT-LLM/requirements-dev.txt (line 13))
      Using cached nltk-3.8.1-py3-none-any.whl.metadata (2.8 kB)
    Collecting aiosignal>=1.1.2 (from aiohttp->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached aiosignal-1.3.1-py3-none-any.whl.metadata (4.0 kB)
    Collecting frozenlist>=1.1.1 (from aiohttp->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached frozenlist-1.4.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
    Collecting multidict<7.0,>=4.5 (from aiohttp->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached multidict-6.0.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)
    Collecting yarl<2.0,>=1.0 (from aiohttp->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached yarl-1.9.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (31 kB)
    Collecting async-timeout<5.0,>=4.0 (from aiohttp->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached async_timeout-4.0.3-py3-none-any.whl.metadata (4.2 kB)
    Collecting annotated-types>=0.4.0 (from pydantic>=2.0->nvidia-modelopt<0.14,~=0.13->-r TensorRT-LLM/requirements.txt (line 24))
      Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
    Collecting pydantic-core==2.20.1 (from pydantic>=2.0->nvidia-modelopt<0.14,~=0.13->-r TensorRT-LLM/requirements.txt (line 24))
      Using cached pydantic_core-2.20.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
    Collecting iniconfig (from pytest>=4.6->pytest-cov->-r TensorRT-LLM/requirements-dev.txt (line 10))
      Using cached iniconfig-2.0.0-py3-none-any.whl.metadata (2.6 kB)
    Collecting pluggy<2.0,>=1.5 (from pytest>=4.6->pytest-cov->-r TensorRT-LLM/requirements-dev.txt (line 10))
      Using cached pluggy-1.5.0-py3-none-any.whl.metadata (4.8 kB)
    Collecting exceptiongroup>=1.0.0rc8 (from pytest>=4.6->pytest-cov->-r TensorRT-LLM/requirements-dev.txt (line 10))
      Using cached exceptiongroup-1.2.2-py3-none-any.whl.metadata (6.6 kB)
    Collecting charset-normalizer<4,>=2 (from requests>=2.32.1->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
    Collecting idna<4,>=2.5 (from requests>=2.32.1->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached idna-3.7-py3-none-any.whl.metadata (9.9 kB)
    Collecting urllib3<3,>=1.21.1 (from requests>=2.32.1->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached urllib3-2.2.2-py3-none-any.whl.metadata (6.4 kB)
    Collecting certifi>=2017.4.17 (from requests>=2.32.1->datasets==2.19.2->-r TensorRT-LLM/requirements-dev.txt (line 2))
      Using cached certifi-2024.7.4-py3-none-any.whl.metadata (2.2 kB)
    Collecting pbr!=2.1.0,>=2.0.0 (from stevedore>=1.20.0->bandit==1.7.7->-r TensorRT-LLM/requirements-dev.txt (line 16))
      Using cached pbr-6.0.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting distlib<1,>=0.3.7 (from virtualenv>=20.10.0->pre-commit->-r TensorRT-LLM/requirements-dev.txt (line 7))
      Using cached distlib-0.3.8-py2.py3-none-any.whl.metadata (5.1 kB)
    Collecting platformdirs<5,>=3.9.1 (from virtualenv>=20.10.0->pre-commit->-r TensorRT-LLM/requirements-dev.txt (line 7))
      Using cached platformdirs-4.2.2-py3-none-any.whl.metadata (11 kB)
    Collecting humanfriendly>=9.1 (from coloredlogs->optimum->-r TensorRT-LLM/requirements.txt (line 28))
      Using cached humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
    Collecting zipp>=0.5 (from importlib-metadata->diffusers>=0.27.0->-r TensorRT-LLM/requirements.txt (line 6))
      Using cached zipp-3.19.2-py3-none-any.whl.metadata (3.6 kB)
    Collecting MarkupSafe>=2.0 (from jinja2->torch<=2.4.0a0,>=2.3.0a0->-r TensorRT-LLM/requirements.txt (line 23))
      Using cached MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
    Collecting click (from nltk->rouge_score->-r TensorRT-LLM/requirements-dev.txt (line 13))
      Using cached click-8.1.7-py3-none-any.whl.metadata (3.0 kB)
    Collecting joblib (from nltk->rouge_score->-r TensorRT-LLM/requirements-dev.txt (line 13))
      Using cached joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting markdown-it-py>=2.2.0 (from rich->bandit==1.7.7->-r TensorRT-LLM/requirements-dev.txt (line 16))
      Using cached markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
    Collecting pygments<3.0.0,>=2.13.0 (from rich->bandit==1.7.7->-r TensorRT-LLM/requirements-dev.txt (line 16))
      Using cached pygments-2.18.0-py3-none-any.whl.metadata (2.5 kB)
    Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->bandit==1.7.7->-r TensorRT-LLM/requirements-dev.txt (line 16))
      Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
    Using cached h5py-3.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.8 MB)
    Using cached pillow-10.3.0-cp310-cp310-manylinux_2_28_x86_64.whl (4.5 MB)
    Using cached datasets-2.19.2-py3-none-any.whl (542 kB)
    Using cached typing_extensions-4.8.0-py3-none-any.whl (31 kB)
    Using cached bandit-1.7.7-py3-none-any.whl (124 kB)
    Using cached jsonlines-4.0.0-py3-none-any.whl (8.7 kB)
    Using cached rouge-1.0.1-py3-none-any.whl (13 kB)
    Using cached accelerate-0.32.1-py3-none-any.whl (314 kB)
    Using cached build-1.2.1-py3-none-any.whl (21 kB)
    Using cached colored-2.2.4-py3-none-any.whl (16 kB)
    Using cached cuda_python-12.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (24.8 MB)
    Using cached diffusers-0.29.2-py3-none-any.whl (2.2 MB)
    Using cached lark-1.1.9-py3-none-any.whl (111 kB)
    Using cached numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
    Using cached onnx-1.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (15.9 MB)
    Using cached psutil-6.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (290 kB)
    Using cached pynvml-11.5.2-py3-none-any.whl (53 kB)
    Using cached PuLP-2.9.0-py3-none-any.whl (17.7 MB)
    Using cached pandas-2.2.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.0 MB)
    Using cached StrEnum-0.4.15-py3-none-any.whl (8.9 kB)
    Using cached sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    Using cached torch-2.3.1-cp310-cp310-manylinux1_x86_64.whl (779.1 MB)
    Using cached triton-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (168.1 MB)
    Using cached transformers-4.42.4-py3-none-any.whl (9.3 MB)
    Using cached wheel-0.43.0-py3-none-any.whl (65 kB)
    Using cached optimum-1.21.2-py3-none-any.whl (424 kB)
    Using cached evaluate-0.4.2-py3-none-any.whl (84 kB)
    Using cached janus-1.0.0-py3-none-any.whl (6.9 kB)
    Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
    Using cached einops-0.8.0-py3-none-any.whl (43 kB)
    Using cached graphviz-0.20.3-py3-none-any.whl (47 kB)
    Using cached mypy-1.10.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.7 MB)
    Using cached parameterized-0.9.0-py2.py3-none-any.whl (20 kB)
    Using cached pre_commit-3.7.1-py2.py3-none-any.whl (204 kB)
    Using cached pybind11-2.13.1-py3-none-any.whl (238 kB)
    Using cached pybind11_stubgen-2.5.1-py3-none-any.whl (29 kB)
    Using cached pytest_cov-5.0.0-py3-none-any.whl (21 kB)
    Using cached pytest_forked-1.6.0-py3-none-any.whl (4.9 kB)
    Using cached pytest_xdist-3.6.1-py3-none-any.whl (46 kB)
    Using cached cloudpickle-3.0.0-py3-none-any.whl (20 kB)
    Using cached attrs-23.2.0-py3-none-any.whl (60 kB)
    Using cached cfgv-3.4.0-py2.py3-none-any.whl (7.2 kB)
    Using cached coverage-7.6.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (233 kB)
    Using cached dill-0.3.8-py3-none-any.whl (116 kB)
    Using cached execnet-2.1.1-py3-none-any.whl (40 kB)
    Using cached fsspec-2024.3.1-py3-none-any.whl (171 kB)
    Using cached aiohttp-3.9.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
    Using cached huggingface_hub-0.23.4-py3-none-any.whl (402 kB)
    Using cached identify-2.6.0-py2.py3-none-any.whl (98 kB)
    Using cached mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)
    Using cached nodeenv-1.9.1-py2.py3-none-any.whl (22 kB)
    Using cached packaging-24.1-py3-none-any.whl (53 kB)
    Using cached protobuf-5.27.2-cp38-abi3-manylinux2014_x86_64.whl (309 kB)
    Using cached pyarrow-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl (40.8 MB)
    Using cached pydantic-2.8.2-py3-none-any.whl (423 kB)
    Using cached pydantic_core-2.20.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
    Using cached pytest-8.2.2-py3-none-any.whl (339 kB)
    Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
    Using cached pytz-2024.1-py2.py3-none-any.whl (505 kB)
    Using cached PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (705 kB)
    Using cached regex-2024.5.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (775 kB)
    Using cached requests-2.32.3-py3-none-any.whl (64 kB)
    Using cached safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
    Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
    Using cached stevedore-5.2.0-py3-none-any.whl (49 kB)
    Using cached tokenizers-0.19.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.6 MB)
    Using cached tomli-2.0.1-py3-none-any.whl (12 kB)
    Using cached tqdm-4.66.4-py3-none-any.whl (78 kB)
    Using cached tzdata-2024.1-py2.py3-none-any.whl (345 kB)
    Using cached virtualenv-20.26.3-py3-none-any.whl (5.7 MB)
    Using cached filelock-3.15.4-py3-none-any.whl (16 kB)
    Using cached absl_py-2.1.0-py3-none-any.whl (133 kB)
    Using cached coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
    Using cached importlib_metadata-8.0.0-py3-none-any.whl (24 kB)
    Using cached jinja2-3.1.4-py3-none-any.whl (133 kB)
    Using cached multiprocess-0.70.16-py310-none-any.whl (134 kB)
    Using cached networkx-3.3-py3-none-any.whl (1.7 MB)
    Using cached ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (307 kB)
    Using cached nltk-3.8.1-py3-none-any.whl (1.5 MB)
    Using cached py-1.11.0-py2.py3-none-any.whl (98 kB)
    Using cached pyarrow_hotfix-0.6-py3-none-any.whl (7.9 kB)
    Using cached pyproject_hooks-1.1.0-py3-none-any.whl (9.2 kB)
    Using cached rich-13.7.1-py3-none-any.whl (240 kB)
    Using cached scipy-1.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (41.1 MB)
    Using cached sympy-1.13.0-py3-none-any.whl (6.2 MB)
    Using cached xxhash-3.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)
    Using cached aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
    Using cached async_timeout-4.0.3-py3-none-any.whl (5.7 kB)
    Using cached certifi-2024.7.4-py3-none-any.whl (162 kB)
    Using cached charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
    Using cached distlib-0.3.8-py2.py3-none-any.whl (468 kB)
    Using cached exceptiongroup-1.2.2-py3-none-any.whl (16 kB)
    Using cached frozenlist-1.4.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (239 kB)
    Using cached humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
    Using cached idna-3.7-py3-none-any.whl (66 kB)
    Using cached markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
    Using cached MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
    Using cached multidict-6.0.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (124 kB)
    Using cached pbr-6.0.0-py2.py3-none-any.whl (107 kB)
    Using cached platformdirs-4.2.2-py3-none-any.whl (18 kB)
    Using cached pluggy-1.5.0-py3-none-any.whl (20 kB)
    Using cached pygments-2.18.0-py3-none-any.whl (1.2 MB)
    Using cached urllib3-2.2.2-py3-none-any.whl (121 kB)
    Using cached yarl-1.9.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (301 kB)
    Using cached zipp-3.19.2-py3-none-any.whl (9.0 kB)
    Using cached click-8.1.7-py3-none-any.whl (97 kB)
    Using cached iniconfig-2.0.0-py3-none-any.whl (5.9 kB)
    Using cached joblib-1.4.2-py3-none-any.whl (301 kB)
    Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Installing collected packages: tensorrt-cu12-bindings, StrEnum, sentencepiece, pytz, ninja, mpmath, jieba, distlib, cuda-python, zipp, xxhash, wheel, urllib3, tzdata, typing-extensions, tqdm, tomli, sympy, six, safetensors, regex, pyyaml, pyproject_hooks, pynvml, pygments, pybind11-stubgen, pybind11, pyarrow-hotfix, py, pulp, psutil, protobuf, polygraphy, pluggy, platformdirs, pillow, pbr, parameterized, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, nodeenv, networkx, mypy-extensions, multidict, mpi4py, mdurl, MarkupSafe, lark, joblib, iniconfig, idna, identify, humanfriendly, graphviz, fsspec, frozenlist, filelock, execnet, exceptiongroup, einops, dill, coverage, colored, cloudpickle, click, charset-normalizer, cfgv, certifi, attrs, async-timeout, annotated-types, absl-py, yarl, virtualenv, triton, tensorrt-cu12-libs, stevedore, scipy, rouge, requests, python-dateutil, pytest, pydantic-core, pyarrow, onnx, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nltk, mypy, multiprocess, markdown-it-py, jsonlines, jinja2, janus, importlib-metadata, h5py, coloredlogs, build, aiosignal, tensorrt-cu12, rouge_score, rich, pytest-xdist, pytest-forked, pytest-cov, pydantic, pre-commit, pandas, nvidia-cusolver-cu12, huggingface-hub, aiohttp, torch, tokenizers, nvidia-modelopt, diffusers, bandit, transformers, datasets, accelerate, evaluate, optimum
      Attempting uninstall: tensorrt-cu12-bindings
        Found existing installation: tensorrt-cu12-bindings 10.1.0
        Uninstalling tensorrt-cu12-bindings-10.1.0:
          Successfully uninstalled tensorrt-cu12-bindings-10.1.0
      Attempting uninstall: StrEnum
        Found existing installation: StrEnum 0.4.15
        Uninstalling StrEnum-0.4.15:
          Successfully uninstalled StrEnum-0.4.15
      Attempting uninstall: sentencepiece
        Found existing installation: sentencepiece 0.2.0
        Uninstalling sentencepiece-0.2.0:
          Successfully uninstalled sentencepiece-0.2.0
      Attempting uninstall: pytz
        Found existing installation: pytz 2024.1
        Uninstalling pytz-2024.1:
          Successfully uninstalled pytz-2024.1
      Attempting uninstall: ninja
        Found existing installation: ninja 1.11.1.1
        Uninstalling ninja-1.11.1.1:
          Successfully uninstalled ninja-1.11.1.1
      Attempting uninstall: mpmath
        Found existing installation: mpmath 1.3.0
        Uninstalling mpmath-1.3.0:
          Successfully uninstalled mpmath-1.3.0
      Attempting uninstall: jieba
        Found existing installation: jieba 0.42.1
        Uninstalling jieba-0.42.1:
          Successfully uninstalled jieba-0.42.1
      Attempting uninstall: distlib
        Found existing installation: distlib 0.3.8
        Uninstalling distlib-0.3.8:
          Successfully uninstalled distlib-0.3.8
      Attempting uninstall: cuda-python
        Found existing installation: cuda-python 12.5.0
        Uninstalling cuda-python-12.5.0:
          Successfully uninstalled cuda-python-12.5.0
      Attempting uninstall: zipp
        Found existing installation: zipp 3.19.2
        Uninstalling zipp-3.19.2:
          Successfully uninstalled zipp-3.19.2
      Attempting uninstall: xxhash
        Found existing installation: xxhash 3.4.1
        Uninstalling xxhash-3.4.1:
          Successfully uninstalled xxhash-3.4.1
      Attempting uninstall: wheel
        Found existing installation: wheel 0.43.0
        Uninstalling wheel-0.43.0:
          Successfully uninstalled wheel-0.43.0
      Attempting uninstall: urllib3
        Found existing installation: urllib3 2.2.2
        Uninstalling urllib3-2.2.2:
          Successfully uninstalled urllib3-2.2.2
      Attempting uninstall: tzdata
        Found existing installation: tzdata 2024.1
        Uninstalling tzdata-2024.1:
          Successfully uninstalled tzdata-2024.1
      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.8.0
        Uninstalling typing_extensions-4.8.0:
          Successfully uninstalled typing_extensions-4.8.0
      Attempting uninstall: tqdm
        Found existing installation: tqdm 4.66.4
        Uninstalling tqdm-4.66.4:
          Successfully uninstalled tqdm-4.66.4
      Attempting uninstall: tomli
        Found existing installation: tomli 2.0.1
        Uninstalling tomli-2.0.1:
          Successfully uninstalled tomli-2.0.1
      Attempting uninstall: sympy
        Found existing installation: sympy 1.13.0
        Uninstalling sympy-1.13.0:
          Successfully uninstalled sympy-1.13.0
      Attempting uninstall: six
        Found existing installation: six 1.16.0
        Uninstalling six-1.16.0:
          Successfully uninstalled six-1.16.0
      Attempting uninstall: safetensors
        Found existing installation: safetensors 0.4.3
        Uninstalling safetensors-0.4.3:
          Successfully uninstalled safetensors-0.4.3
      Attempting uninstall: regex
        Found existing installation: regex 2024.5.15
        Uninstalling regex-2024.5.15:
          Successfully uninstalled regex-2024.5.15
      Attempting uninstall: pyyaml
        Found existing installation: PyYAML 6.0.1
        Uninstalling PyYAML-6.0.1:
          Successfully uninstalled PyYAML-6.0.1
      Attempting uninstall: pyproject_hooks
        Found existing installation: pyproject_hooks 1.1.0
        Uninstalling pyproject_hooks-1.1.0:
          Successfully uninstalled pyproject_hooks-1.1.0
      Attempting uninstall: pynvml
        Found existing installation: pynvml 11.5.2
        Uninstalling pynvml-11.5.2:
          Successfully uninstalled pynvml-11.5.2
      Attempting uninstall: pygments
        Found existing installation: Pygments 2.18.0
        Uninstalling Pygments-2.18.0:
          Successfully uninstalled Pygments-2.18.0
      Attempting uninstall: pybind11-stubgen
        Found existing installation: pybind11-stubgen 2.5.1
        Uninstalling pybind11-stubgen-2.5.1:
          Successfully uninstalled pybind11-stubgen-2.5.1
      Attempting uninstall: pybind11
        Found existing installation: pybind11 2.13.1
        Uninstalling pybind11-2.13.1:
          Successfully uninstalled pybind11-2.13.1
      Attempting uninstall: pyarrow-hotfix
        Found existing installation: pyarrow-hotfix 0.6
        Uninstalling pyarrow-hotfix-0.6:
          Successfully uninstalled pyarrow-hotfix-0.6
      Attempting uninstall: py
        Found existing installation: py 1.11.0
        Uninstalling py-1.11.0:
          Successfully uninstalled py-1.11.0
      Attempting uninstall: pulp
        Found existing installation: PuLP 2.9.0
        Uninstalling PuLP-2.9.0:
          Successfully uninstalled PuLP-2.9.0
      Attempting uninstall: psutil
        Found existing installation: psutil 6.0.0
        Uninstalling psutil-6.0.0:
          Successfully uninstalled psutil-6.0.0
      Attempting uninstall: protobuf
        Found existing installation: protobuf 5.27.2
        Uninstalling protobuf-5.27.2:
          Successfully uninstalled protobuf-5.27.2
      Attempting uninstall: polygraphy
        Found existing installation: polygraphy 0.49.9
        Uninstalling polygraphy-0.49.9:
          Successfully uninstalled polygraphy-0.49.9
      Attempting uninstall: pluggy
        Found existing installation: pluggy 1.5.0
        Uninstalling pluggy-1.5.0:
          Successfully uninstalled pluggy-1.5.0
      Attempting uninstall: platformdirs
        Found existing installation: platformdirs 4.2.2
        Uninstalling platformdirs-4.2.2:
          Successfully uninstalled platformdirs-4.2.2
      Attempting uninstall: pillow
        Found existing installation: pillow 10.3.0
        Uninstalling pillow-10.3.0:
          Successfully uninstalled pillow-10.3.0
      Attempting uninstall: pbr
        Found existing installation: pbr 6.0.0
        Uninstalling pbr-6.0.0:
          Successfully uninstalled pbr-6.0.0
      Attempting uninstall: parameterized
        Found existing installation: parameterized 0.9.0
        Uninstalling parameterized-0.9.0:
          Successfully uninstalled parameterized-0.9.0
      Attempting uninstall: packaging
        Found existing installation: packaging 24.1
        Uninstalling packaging-24.1:
          Successfully uninstalled packaging-24.1
      Attempting uninstall: nvidia-nvtx-cu12
        Found existing installation: nvidia-nvtx-cu12 12.1.105
        Uninstalling nvidia-nvtx-cu12-12.1.105:
          Successfully uninstalled nvidia-nvtx-cu12-12.1.105
      Attempting uninstall: nvidia-nvjitlink-cu12
        Found existing installation: nvidia-nvjitlink-cu12 12.5.82
        Uninstalling nvidia-nvjitlink-cu12-12.5.82:
          Successfully uninstalled nvidia-nvjitlink-cu12-12.5.82
      Attempting uninstall: nvidia-nccl-cu12
        Found existing installation: nvidia-nccl-cu12 2.20.5
        Uninstalling nvidia-nccl-cu12-2.20.5:
          Successfully uninstalled nvidia-nccl-cu12-2.20.5
      Attempting uninstall: nvidia-curand-cu12
        Found existing installation: nvidia-curand-cu12 10.3.2.106
        Uninstalling nvidia-curand-cu12-10.3.2.106:
          Successfully uninstalled nvidia-curand-cu12-10.3.2.106
      Attempting uninstall: nvidia-cufft-cu12
        Found existing installation: nvidia-cufft-cu12 11.0.2.54
        Uninstalling nvidia-cufft-cu12-11.0.2.54:
          Successfully uninstalled nvidia-cufft-cu12-11.0.2.54
      Attempting uninstall: nvidia-cuda-runtime-cu12
        Found existing installation: nvidia-cuda-runtime-cu12 12.1.105
        Uninstalling nvidia-cuda-runtime-cu12-12.1.105:
          Successfully uninstalled nvidia-cuda-runtime-cu12-12.1.105
      Attempting uninstall: nvidia-cuda-nvrtc-cu12
        Found existing installation: nvidia-cuda-nvrtc-cu12 12.1.105
        Uninstalling nvidia-cuda-nvrtc-cu12-12.1.105:
          Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.1.105
      Attempting uninstall: nvidia-cuda-cupti-cu12
        Found existing installation: nvidia-cuda-cupti-cu12 12.1.105
        Uninstalling nvidia-cuda-cupti-cu12-12.1.105:
          Successfully uninstalled nvidia-cuda-cupti-cu12-12.1.105
      Attempting uninstall: nvidia-cublas-cu12
        Found existing installation: nvidia-cublas-cu12 12.1.3.1
        Uninstalling nvidia-cublas-cu12-12.1.3.1:
          Successfully uninstalled nvidia-cublas-cu12-12.1.3.1
      Attempting uninstall: numpy
        Found existing installation: numpy 1.26.4
        Uninstalling numpy-1.26.4:
          Successfully uninstalled numpy-1.26.4
      Attempting uninstall: nodeenv
        Found existing installation: nodeenv 1.9.1
        Uninstalling nodeenv-1.9.1:
          Successfully uninstalled nodeenv-1.9.1
      Attempting uninstall: networkx
        Found existing installation: networkx 3.3
        Uninstalling networkx-3.3:
          Successfully uninstalled networkx-3.3
      Attempting uninstall: mypy-extensions
        Found existing installation: mypy-extensions 1.0.0
        Uninstalling mypy-extensions-1.0.0:
          Successfully uninstalled mypy-extensions-1.0.0
      Attempting uninstall: multidict
        Found existing installation: multidict 6.0.5
        Uninstalling multidict-6.0.5:
          Successfully uninstalled multidict-6.0.5
      Attempting uninstall: mpi4py
        Found existing installation: mpi4py 3.1.6
        Uninstalling mpi4py-3.1.6:
          Successfully uninstalled mpi4py-3.1.6
      Attempting uninstall: mdurl
        Found existing installation: mdurl 0.1.2
        Uninstalling mdurl-0.1.2:
          Successfully uninstalled mdurl-0.1.2
      Attempting uninstall: MarkupSafe
        Found existing installation: MarkupSafe 2.1.5
        Uninstalling MarkupSafe-2.1.5:
          Successfully uninstalled MarkupSafe-2.1.5
      Attempting uninstall: lark
        Found existing installation: lark 1.1.9
        Uninstalling lark-1.1.9:
          Successfully uninstalled lark-1.1.9
      Attempting uninstall: joblib
        Found existing installation: joblib 1.4.2
        Uninstalling joblib-1.4.2:
          Successfully uninstalled joblib-1.4.2
      Attempting uninstall: iniconfig
        Found existing installation: iniconfig 2.0.0
        Uninstalling iniconfig-2.0.0:
          Successfully uninstalled iniconfig-2.0.0
      Attempting uninstall: idna
        Found existing installation: idna 3.7
        Uninstalling idna-3.7:
          Successfully uninstalled idna-3.7
      Attempting uninstall: identify
        Found existing installation: identify 2.6.0
        Uninstalling identify-2.6.0:
          Successfully uninstalled identify-2.6.0
      Attempting uninstall: humanfriendly
        Found existing installation: humanfriendly 10.0
        Uninstalling humanfriendly-10.0:
          Successfully uninstalled humanfriendly-10.0
      Attempting uninstall: graphviz
        Found existing installation: graphviz 0.20.3
        Uninstalling graphviz-0.20.3:
          Successfully uninstalled graphviz-0.20.3
      Attempting uninstall: fsspec
        Found existing installation: fsspec 2024.3.1
        Uninstalling fsspec-2024.3.1:
          Successfully uninstalled fsspec-2024.3.1
      Attempting uninstall: frozenlist
        Found existing installation: frozenlist 1.4.1
        Uninstalling frozenlist-1.4.1:
          Successfully uninstalled frozenlist-1.4.1
      Attempting uninstall: filelock
        Found existing installation: filelock 3.15.4
        Uninstalling filelock-3.15.4:
          Successfully uninstalled filelock-3.15.4
      Attempting uninstall: execnet
        Found existing installation: execnet 2.1.1
        Uninstalling execnet-2.1.1:
          Successfully uninstalled execnet-2.1.1
      Attempting uninstall: exceptiongroup
        Found existing installation: exceptiongroup 1.2.2
        Uninstalling exceptiongroup-1.2.2:
          Successfully uninstalled exceptiongroup-1.2.2
      Attempting uninstall: einops
        Found existing installation: einops 0.8.0
        Uninstalling einops-0.8.0:
          Successfully uninstalled einops-0.8.0
      Attempting uninstall: dill
        Found existing installation: dill 0.3.8
        Uninstalling dill-0.3.8:
          Successfully uninstalled dill-0.3.8
      Attempting uninstall: coverage
        Found existing installation: coverage 7.6.0
        Uninstalling coverage-7.6.0:
          Successfully uninstalled coverage-7.6.0
      Attempting uninstall: colored
        Found existing installation: colored 2.2.4
        Uninstalling colored-2.2.4:
          Successfully uninstalled colored-2.2.4
      Attempting uninstall: cloudpickle
        Found existing installation: cloudpickle 3.0.0
        Uninstalling cloudpickle-3.0.0:
          Successfully uninstalled cloudpickle-3.0.0
      Attempting uninstall: click
        Found existing installation: click 8.1.7
        Uninstalling click-8.1.7:
          Successfully uninstalled click-8.1.7
      Attempting uninstall: charset-normalizer
        Found existing installation: charset-normalizer 3.3.2
        Uninstalling charset-normalizer-3.3.2:
          Successfully uninstalled charset-normalizer-3.3.2
      Attempting uninstall: cfgv
        Found existing installation: cfgv 3.4.0
        Uninstalling cfgv-3.4.0:
          Successfully uninstalled cfgv-3.4.0
      Attempting uninstall: certifi
        Found existing installation: certifi 2024.7.4
        Uninstalling certifi-2024.7.4:
          Successfully uninstalled certifi-2024.7.4
      Attempting uninstall: attrs
        Found existing installation: attrs 23.2.0
        Uninstalling attrs-23.2.0:
          Successfully uninstalled attrs-23.2.0
      Attempting uninstall: async-timeout
        Found existing installation: async-timeout 4.0.3
        Uninstalling async-timeout-4.0.3:
          Successfully uninstalled async-timeout-4.0.3
      Attempting uninstall: annotated-types
        Found existing installation: annotated-types 0.7.0
        Uninstalling annotated-types-0.7.0:
          Successfully uninstalled annotated-types-0.7.0
      Attempting uninstall: absl-py
        Found existing installation: absl-py 2.1.0
        Uninstalling absl-py-2.1.0:
          Successfully uninstalled absl-py-2.1.0
      Attempting uninstall: yarl
        Found existing installation: yarl 1.9.4
        Uninstalling yarl-1.9.4:
          Successfully uninstalled yarl-1.9.4
      Attempting uninstall: virtualenv
        Found existing installation: virtualenv 20.26.3
        Uninstalling virtualenv-20.26.3:
          Successfully uninstalled virtualenv-20.26.3
      Attempting uninstall: triton
        Found existing installation: triton 2.3.1
        Uninstalling triton-2.3.1:
          Successfully uninstalled triton-2.3.1
      Attempting uninstall: tensorrt-cu12-libs
        Found existing installation: tensorrt-cu12-libs 10.1.0
        Uninstalling tensorrt-cu12-libs-10.1.0:
          Successfully uninstalled tensorrt-cu12-libs-10.1.0
      Attempting uninstall: stevedore
        Found existing installation: stevedore 5.2.0
        Uninstalling stevedore-5.2.0:
          Successfully uninstalled stevedore-5.2.0
      Attempting uninstall: scipy
        Found existing installation: scipy 1.14.0
        Uninstalling scipy-1.14.0:
          Successfully uninstalled scipy-1.14.0
      Attempting uninstall: rouge
        Found existing installation: rouge 1.0.1
        Uninstalling rouge-1.0.1:
          Successfully uninstalled rouge-1.0.1
      Attempting uninstall: requests
        Found existing installation: requests 2.32.3
        Uninstalling requests-2.32.3:
          Successfully uninstalled requests-2.32.3
      Attempting uninstall: python-dateutil
        Found existing installation: python-dateutil 2.9.0.post0
        Uninstalling python-dateutil-2.9.0.post0:
          Successfully uninstalled python-dateutil-2.9.0.post0
      Attempting uninstall: pytest
        Found existing installation: pytest 8.2.2
        Uninstalling pytest-8.2.2:
          Successfully uninstalled pytest-8.2.2
      Attempting uninstall: pydantic-core
        Found existing installation: pydantic_core 2.20.1
        Uninstalling pydantic_core-2.20.1:
          Successfully uninstalled pydantic_core-2.20.1
      Attempting uninstall: pyarrow
        Found existing installation: pyarrow 16.1.0
        Uninstalling pyarrow-16.1.0:
          Successfully uninstalled pyarrow-16.1.0
      Attempting uninstall: onnx
        Found existing installation: onnx 1.16.1
        Uninstalling onnx-1.16.1:
          Successfully uninstalled onnx-1.16.1
      Attempting uninstall: nvidia-cusparse-cu12
        Found existing installation: nvidia-cusparse-cu12 12.1.0.106
        Uninstalling nvidia-cusparse-cu12-12.1.0.106:
          Successfully uninstalled nvidia-cusparse-cu12-12.1.0.106
      Attempting uninstall: nvidia-cudnn-cu12
        Found existing installation: nvidia-cudnn-cu12 8.9.2.26
        Uninstalling nvidia-cudnn-cu12-8.9.2.26:
          Successfully uninstalled nvidia-cudnn-cu12-8.9.2.26
      Attempting uninstall: nltk
        Found existing installation: nltk 3.8.1
        Uninstalling nltk-3.8.1:
          Successfully uninstalled nltk-3.8.1
      Attempting uninstall: mypy
        Found existing installation: mypy 1.10.1
        Uninstalling mypy-1.10.1:
          Successfully uninstalled mypy-1.10.1
      Attempting uninstall: multiprocess
        Found existing installation: multiprocess 0.70.16
        Uninstalling multiprocess-0.70.16:
          Successfully uninstalled multiprocess-0.70.16
      Attempting uninstall: markdown-it-py
        Found existing installation: markdown-it-py 3.0.0
        Uninstalling markdown-it-py-3.0.0:
          Successfully uninstalled markdown-it-py-3.0.0
      Attempting uninstall: jsonlines
        Found existing installation: jsonlines 4.0.0
        Uninstalling jsonlines-4.0.0:
          Successfully uninstalled jsonlines-4.0.0
      Attempting uninstall: jinja2
        Found existing installation: Jinja2 3.1.4
        Uninstalling Jinja2-3.1.4:
          Successfully uninstalled Jinja2-3.1.4
      Attempting uninstall: janus
        Found existing installation: janus 1.0.0
        Uninstalling janus-1.0.0:
          Successfully uninstalled janus-1.0.0
      Attempting uninstall: importlib-metadata
        Found existing installation: importlib_metadata 8.0.0
        Uninstalling importlib_metadata-8.0.0:
          Successfully uninstalled importlib_metadata-8.0.0
      Attempting uninstall: h5py
        Found existing installation: h5py 3.10.0
        Uninstalling h5py-3.10.0:
          Successfully uninstalled h5py-3.10.0
      Attempting uninstall: coloredlogs
        Found existing installation: coloredlogs 15.0.1
        Uninstalling coloredlogs-15.0.1:
          Successfully uninstalled coloredlogs-15.0.1
      Attempting uninstall: build
        Found existing installation: build 1.2.1
        Uninstalling build-1.2.1:
          Successfully uninstalled build-1.2.1
      Attempting uninstall: aiosignal
        Found existing installation: aiosignal 1.3.1
        Uninstalling aiosignal-1.3.1:
          Successfully uninstalled aiosignal-1.3.1
      Attempting uninstall: tensorrt-cu12
        Found existing installation: tensorrt-cu12 10.1.0
        Uninstalling tensorrt-cu12-10.1.0:
          Successfully uninstalled tensorrt-cu12-10.1.0
      Attempting uninstall: rouge_score
        Found existing installation: rouge_score 0.1.2
        Uninstalling rouge_score-0.1.2:
          Successfully uninstalled rouge_score-0.1.2
      Attempting uninstall: rich
        Found existing installation: rich 13.7.1
        Uninstalling rich-13.7.1:
          Successfully uninstalled rich-13.7.1
      Attempting uninstall: pytest-xdist
        Found existing installation: pytest-xdist 3.6.1
        Uninstalling pytest-xdist-3.6.1:
          Successfully uninstalled pytest-xdist-3.6.1
      Attempting uninstall: pytest-forked
        Found existing installation: pytest-forked 1.6.0
        Uninstalling pytest-forked-1.6.0:
          Successfully uninstalled pytest-forked-1.6.0
      Attempting uninstall: pytest-cov
        Found existing installation: pytest-cov 5.0.0
        Uninstalling pytest-cov-5.0.0:
          Successfully uninstalled pytest-cov-5.0.0
      Attempting uninstall: pydantic
        Found existing installation: pydantic 2.8.2
        Uninstalling pydantic-2.8.2:
          Successfully uninstalled pydantic-2.8.2
      Attempting uninstall: pre-commit
        Found existing installation: pre-commit 3.7.1
        Uninstalling pre-commit-3.7.1:
          Successfully uninstalled pre-commit-3.7.1
      Attempting uninstall: pandas
        Found existing installation: pandas 2.2.2
        Uninstalling pandas-2.2.2:
          Successfully uninstalled pandas-2.2.2
      Attempting uninstall: nvidia-cusolver-cu12
        Found existing installation: nvidia-cusolver-cu12 11.4.5.107
        Uninstalling nvidia-cusolver-cu12-11.4.5.107:
          Successfully uninstalled nvidia-cusolver-cu12-11.4.5.107
      Attempting uninstall: huggingface-hub
        Found existing installation: huggingface-hub 0.23.4
        Uninstalling huggingface-hub-0.23.4:
          Successfully uninstalled huggingface-hub-0.23.4
      Attempting uninstall: aiohttp
        Found existing installation: aiohttp 3.9.5
        Uninstalling aiohttp-3.9.5:
          Successfully uninstalled aiohttp-3.9.5
      Attempting uninstall: torch
        Found existing installation: torch 2.3.1
        Uninstalling torch-2.3.1:
          Successfully uninstalled torch-2.3.1
      Attempting uninstall: tokenizers
        Found existing installation: tokenizers 0.19.1
        Uninstalling tokenizers-0.19.1:
          Successfully uninstalled tokenizers-0.19.1
      Attempting uninstall: nvidia-modelopt
        Found existing installation: nvidia-modelopt 0.13.1
        Uninstalling nvidia-modelopt-0.13.1:
          Successfully uninstalled nvidia-modelopt-0.13.1
      Attempting uninstall: diffusers
        Found existing installation: diffusers 0.29.2
        Uninstalling diffusers-0.29.2:
          Successfully uninstalled diffusers-0.29.2
      Attempting uninstall: bandit
        Found existing installation: bandit 1.7.7
        Uninstalling bandit-1.7.7:
          Successfully uninstalled bandit-1.7.7
      Attempting uninstall: transformers
        Found existing installation: transformers 4.42.4
        Uninstalling transformers-4.42.4:
          Successfully uninstalled transformers-4.42.4
      Attempting uninstall: datasets
        Found existing installation: datasets 2.19.2
        Uninstalling datasets-2.19.2:
          Successfully uninstalled datasets-2.19.2
      Attempting uninstall: accelerate
        Found existing installation: accelerate 0.32.1
        Uninstalling accelerate-0.32.1:
          Successfully uninstalled accelerate-0.32.1
      Attempting uninstall: evaluate
        Found existing installation: evaluate 0.4.2
        Uninstalling evaluate-0.4.2:
          Successfully uninstalled evaluate-0.4.2
      Attempting uninstall: optimum
        Found existing installation: optimum 1.21.2
        Uninstalling optimum-1.21.2:
          Successfully uninstalled optimum-1.21.2
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    cucim 24.4.0 requires cupy-cuda11x>=12.0.0, which is not installed.
    cudf 24.4.1 requires cubinlinker, which is not installed.
    cudf 24.4.1 requires cupy-cuda11x>=12.0.0, which is not installed.
    cudf 24.4.1 requires ptxcompiler, which is not installed.
    cugraph 24.4.0 requires cupy-cuda11x>=12.0.0, which is not installed.
    cuml 24.4.0 requires cupy-cuda11x>=12.0.0, which is not installed.
    cuxfilter 24.4.1 requires cupy-cuda11x>=12.0.0, which is not installed.
    dask-cudf 24.4.1 requires cupy-cuda11x>=12.0.0, which is not installed.
    nx-cugraph 24.4.0 requires cupy-cuda11x>=12.0.0, which is not installed.
    cudf 24.4.1 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.5.0 which is incompatible.
    cudf 24.4.1 requires pandas<2.2.2dev0,>=2.0, but you have pandas 2.2.2 which is incompatible.
    cudf 24.4.1 requires protobuf<5,>=3.20, but you have protobuf 5.27.2 which is incompatible.
    cudf 24.4.1 requires pyarrow<15.0.0a0,>=14.0.1, but you have pyarrow 16.1.0 which is incompatible.
    dask-cuda 24.4.0 requires pynvml<11.5,>=11.0.0, but you have pynvml 11.5.2 which is incompatible.
    dask-cudf 24.4.1 requires pandas<2.2.2dev0,>=2.0, but you have pandas 2.2.2 which is incompatible.
    pylibraft 24.4.0 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.5.0 which is incompatible.
    pyppeteer 1.0.2 requires urllib3<2.0.0,>=1.25.8, but you have urllib3 2.2.2 which is incompatible.
    rmm 24.4.0 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.5.0 which is incompatible.[0m[31m
    [0mSuccessfully installed MarkupSafe-2.1.5 StrEnum-0.4.15 absl-py-2.1.0 accelerate-0.32.1 aiohttp-3.9.5 aiosignal-1.3.1 annotated-types-0.7.0 async-timeout-4.0.3 attrs-23.2.0 bandit-1.7.7 build-1.2.1 certifi-2024.7.4 cfgv-3.4.0 charset-normalizer-3.3.2 click-8.1.7 cloudpickle-3.0.0 colored-2.2.4 coloredlogs-15.0.1 coverage-7.6.0 cuda-python-12.5.0 datasets-2.19.2 diffusers-0.29.2 dill-0.3.8 distlib-0.3.8 einops-0.8.0 evaluate-0.4.2 exceptiongroup-1.2.2 execnet-2.1.1 filelock-3.15.4 frozenlist-1.4.1 fsspec-2024.3.1 graphviz-0.20.3 h5py-3.10.0 huggingface-hub-0.23.4 humanfriendly-10.0 identify-2.6.0 idna-3.7 importlib-metadata-8.0.0 iniconfig-2.0.0 janus-1.0.0 jieba-0.42.1 jinja2-3.1.4 joblib-1.4.2 jsonlines-4.0.0 lark-1.1.9 markdown-it-py-3.0.0 mdurl-0.1.2 mpi4py-3.1.6 mpmath-1.3.0 multidict-6.0.5 multiprocess-0.70.16 mypy-1.10.1 mypy-extensions-1.0.0 networkx-3.3 ninja-1.11.1.1 nltk-3.8.1 nodeenv-1.9.1 numpy-1.26.4 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-modelopt-0.13.1 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.5.82 nvidia-nvtx-cu12-12.1.105 onnx-1.16.1 optimum-1.21.2 packaging-24.1 pandas-2.2.2 parameterized-0.9.0 pbr-6.0.0 pillow-10.3.0 platformdirs-4.2.2 pluggy-1.5.0 polygraphy-0.49.9 pre-commit-3.7.1 protobuf-5.27.2 psutil-6.0.0 pulp-2.9.0 py-1.11.0 pyarrow-16.1.0 pyarrow-hotfix-0.6 pybind11-2.13.1 pybind11-stubgen-2.5.1 pydantic-2.8.2 pydantic-core-2.20.1 pygments-2.18.0 pynvml-11.5.2 pyproject_hooks-1.1.0 pytest-8.2.2 pytest-cov-5.0.0 pytest-forked-1.6.0 pytest-xdist-3.6.1 python-dateutil-2.9.0.post0 pytz-2024.1 pyyaml-6.0.1 regex-2024.5.15 requests-2.32.3 rich-13.7.1 rouge-1.0.1 rouge_score-0.1.2 safetensors-0.4.3 scipy-1.14.0 sentencepiece-0.2.0 six-1.16.0 stevedore-5.2.0 sympy-1.13.0 tensorrt-cu12-10.1.0 tensorrt-cu12-bindings-10.1.0 tensorrt-cu12-libs-10.1.0 tokenizers-0.19.1 tomli-2.0.1 torch-2.3.1 tqdm-4.66.4 transformers-4.42.4 triton-2.3.1 typing-extensions-4.8.0 tzdata-2024.1 urllib3-2.2.2 virtualenv-20.26.3 wheel-0.43.0 xxhash-3.4.1 yarl-1.9.4 zipp-3.19.2
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCPU times: user 1.18 s, sys: 336 ms, total: 1.52 s
    Wall time: 2min 28s


  

  

  

# Download Llama-3 8B Instruct Gradient 1048K
### 2.5 minutes
##### https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k


```python
%%time 
# 2.5 minutes

!git-lfs clone https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k/
```

    WARNING: 'git lfs clone' is deprecated and will not be updated
              with new flags from 'git clone'
    
    'git clone' has been updated in upstream Git to have comparable
    speeds to 'git lfs clone'.
    Cloning into 'Llama-3-8B-Instruct-Gradient-1048k'...
    remote: Enumerating objects: 96, done.[K
    remote: Counting objects: 100% (93/93), done.[K
    remote: Compressing objects: 100% (93/93), done.[K
    remote: Total 96 (delta 49), reused 0 (delta 0), pack-reused 3 (from 1)[K
    Unpacking objects: 100% (96/96), 2.27 MiB | 8.39 MiB/s, done.
    CPU times: user 1.33 s, sys: 388 ms, total: 1.72 sB/s                           
    Wall time: 2min 39s


  

  

# Convert Checkpoint 
### 12 Minutes
##### output dir --> /home/rapids/notebooks/Llama-3-8B-Instruct-Gradient-1048k/trt_ckpts


```python
%%time

# 12 minutes
!python TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir  Llama-3-8B-Instruct-Gradient-1048k/ \
    --output_dir Llama-3-8B-Instruct-Gradient-1048k/trt_ckpts \
    --dtype float16 \
    --tp_size 4
```

    [TensorRT-LLM] TensorRT-LLM version: 0.12.0.dev2024070900
    0.12.0.dev2024070900
    Total time of converting checkpoints: 00:10:55
    CPU times: user 4.61 s, sys: 937 ms, total: 5.55 s
    Wall time: 11min


  

  

# Build TensorRT-LLM Engine
### 6 minutes
###### output dir --> /home/rapids/notebooks/Llama-3-8B-Instruct-Gradient-1048k/trt_engines


```python
%%time 
# 6 minutes

!python -m tensorrt_llm.commands.build \
            --checkpoint_dir Llama-3-8B-Instruct-Gradient-1048k/trt_ckpts \
            --output_dir Llama-3-8B-Instruct-Gradient-1048k/trt_engines \
            --gemm_plugin float16 \
            --max_num_tokens 4096 \
            --max_input_len 1048566 \
            --max_seq_len 1048576 \
            --use_paged_context_fmha enable \
            --workers 4
```

    [TensorRT-LLM] TensorRT-LLM version: 0.12.0.dev2024070900
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set bert_attention_plugin to auto.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set gpt_attention_plugin to auto.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set gemm_plugin to float16.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set gemm_swiglu_plugin to None.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set nccl_plugin to auto.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set lookup_plugin to None.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set lora_plugin to None.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set moe_plugin to auto.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set mamba_conv1d_plugin to auto.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set context_fmha to True.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set context_fmha_fp32_acc to False.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set paged_kv_cache to True.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set remove_input_padding to True.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set use_custom_all_reduce to True.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set reduce_fusion to False.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set multi_block_mode to False.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set enable_xqa to True.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set tokens_per_block to 64.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set use_paged_context_fmha to True.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set use_fp8_context_fmha to False.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set multiple_profiles to False.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set paged_state to True.
    [07/15/2024-22:49:32] [TRT-LLM] [I] Set streamingllm to False.
    [07/15/2024-22:49:32] [TRT-LLM] [W] remove_input_padding is enabled, while opt_num_tokens is not set, setting to max_batch_size*max_beam_width. 
    
    [07/15/2024-22:49:32] [TRT-LLM] [W] padding removal and fMHA are both enabled, max_input_len is not required and will be ignored
    [07/15/2024-22:50:15] [TRT-LLM] [I] Set dtype to float16.
    [07/15/2024-22:50:15] [TRT] [I] [MemUsageChange] Init CUDA: CPU +13, GPU +0, now: CPU 139, GPU 427 (MiB)
    [07/15/2024-22:50:19] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1917, GPU +354, now: CPU 2203, GPU 781 (MiB)
    [07/15/2024-22:50:19] [TRT] [W] profileSharing0806 is on by default in TensorRT 10.0. This flag is deprecated and has no effect.
    [07/15/2024-22:50:19] [TRT-LLM] [I] Set nccl_plugin to float16.
    [07/15/2024-22:50:19] [TRT-LLM] [I] Set use_custom_all_reduce to True.
    [07/15/2024-22:50:20] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
    [07/15/2024-22:50:20] [TRT] [W] Unused Input: position_ids
    [07/15/2024-22:50:20] [TRT] [W] Detected layernorm nodes in FP16.
    [07/15/2024-22:50:20] [TRT] [W] Running layernorm after self-attention in FP16 may cause overflow. Exporting the model to the latest available ONNX opset (later than opset 17) to use the INormalizationLayer, or forcing layernorm layers to run in FP32 precision can help with preserving accuracy.
    [07/15/2024-22:50:20] [TRT] [W] [RemoveDeadLayers] Input Tensor position_ids is unused or used only at compile-time, but is not being removed.
    [07/15/2024-22:50:20] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
    [07/15/2024-22:50:23] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
    [07/15/2024-22:50:23] [TRT] [I] Detected 15 inputs and 1 output network tensors.
    [07/15/2024-22:51:00] [TRT] [I] Total Host Persistent Memory: 111104
    [07/15/2024-22:51:00] [TRT] [I] Total Device Persistent Memory: 0
    [07/15/2024-22:51:00] [TRT] [I] Total Scratch Memory: 67125248
    [07/15/2024-22:51:00] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 626 steps to complete.
    [07/15/2024-22:51:00] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 36.7382ms to assign 17 blocks to 626 nodes requiring 229267456 bytes.
    [07/15/2024-22:51:00] [TRT] [I] Total Activation Memory: 229266432
    [07/15/2024-22:51:00] [TRT] [I] Total Weights Memory: 5340405760
    [07/15/2024-22:51:00] [TRT] [I] Engine generation completed in 40.4034 seconds.
    [07/15/2024-22:51:00] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 512 MiB, GPU 5093 MiB
    [07/15/2024-22:51:04] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 34184 MiB
    [07/15/2024-22:51:04] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:44
    [07/15/2024-22:51:04] [TRT] [I] Serialized 26 bytes of code generator cache.
    [07/15/2024-22:51:04] [TRT] [I] Serialized 135403 bytes of compilation cache.
    [07/15/2024-22:51:04] [TRT] [I] Serialized 13 timing cache entries
    [07/15/2024-22:51:04] [TRT-LLM] [I] Timing cache serialized to model.cache
    [07/15/2024-22:51:04] [TRT-LLM] [I] Serializing engine to Llama-3-8B-Instruct-Gradient-1048k/trt_engines/rank0.engine...
    [07/15/2024-22:51:10] [TRT-LLM] [I] Engine serialized. Total time: 00:00:06
    [07/15/2024-22:51:53] [TRT-LLM] [I] Set dtype to float16.
    [07/15/2024-22:51:53] [TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 2302, GPU 803 (MiB)
    [07/15/2024-22:51:53] [TRT] [W] profileSharing0806 is on by default in TensorRT 10.0. This flag is deprecated and has no effect.
    [07/15/2024-22:51:53] [TRT-LLM] [I] Set nccl_plugin to float16.
    [07/15/2024-22:51:53] [TRT-LLM] [I] Set use_custom_all_reduce to True.
    [07/15/2024-22:51:54] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
    [07/15/2024-22:51:54] [TRT] [W] Unused Input: position_ids
    [07/15/2024-22:51:54] [TRT] [W] Detected layernorm nodes in FP16.
    [07/15/2024-22:51:54] [TRT] [W] Running layernorm after self-attention in FP16 may cause overflow. Exporting the model to the latest available ONNX opset (later than opset 17) to use the INormalizationLayer, or forcing layernorm layers to run in FP32 precision can help with preserving accuracy.
    [07/15/2024-22:51:54] [TRT] [W] [RemoveDeadLayers] Input Tensor position_ids is unused or used only at compile-time, but is not being removed.
    [07/15/2024-22:51:54] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
    [07/15/2024-22:51:56] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
    [07/15/2024-22:51:56] [TRT] [I] Detected 15 inputs and 1 output network tensors.
    [07/15/2024-22:52:27] [TRT] [I] Total Host Persistent Memory: 111104
    [07/15/2024-22:52:27] [TRT] [I] Total Device Persistent Memory: 0
    [07/15/2024-22:52:27] [TRT] [I] Total Scratch Memory: 67125248
    [07/15/2024-22:52:27] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 626 steps to complete.
    [07/15/2024-22:52:27] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 36.5469ms to assign 17 blocks to 626 nodes requiring 229267456 bytes.
    [07/15/2024-22:52:27] [TRT] [I] Total Activation Memory: 229266432
    [07/15/2024-22:52:27] [TRT] [I] Total Weights Memory: 5340405760
    [07/15/2024-22:52:27] [TRT] [I] Engine generation completed in 33.2685 seconds.
    [07/15/2024-22:52:27] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 512 MiB, GPU 5093 MiB
    [07/15/2024-22:52:30] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 58604 MiB
    [07/15/2024-22:52:30] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:36
    [07/15/2024-22:52:32] [TRT-LLM] [I] Serializing engine to Llama-3-8B-Instruct-Gradient-1048k/trt_engines/rank1.engine...
    [07/15/2024-22:52:38] [TRT-LLM] [I] Engine serialized. Total time: 00:00:06
    [07/15/2024-22:53:26] [TRT-LLM] [I] Set dtype to float16.
    [07/15/2024-22:53:26] [TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 2310, GPU 807 (MiB)
    [07/15/2024-22:53:26] [TRT] [W] profileSharing0806 is on by default in TensorRT 10.0. This flag is deprecated and has no effect.
    [07/15/2024-22:53:26] [TRT-LLM] [I] Set nccl_plugin to float16.
    [07/15/2024-22:53:26] [TRT-LLM] [I] Set use_custom_all_reduce to True.
    [07/15/2024-22:53:26] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
    [07/15/2024-22:53:26] [TRT] [W] Unused Input: position_ids
    [07/15/2024-22:53:26] [TRT] [W] Detected layernorm nodes in FP16.
    [07/15/2024-22:53:26] [TRT] [W] Running layernorm after self-attention in FP16 may cause overflow. Exporting the model to the latest available ONNX opset (later than opset 17) to use the INormalizationLayer, or forcing layernorm layers to run in FP32 precision can help with preserving accuracy.
    [07/15/2024-22:53:26] [TRT] [W] [RemoveDeadLayers] Input Tensor position_ids is unused or used only at compile-time, but is not being removed.
    [07/15/2024-22:53:26] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
    [07/15/2024-22:53:29] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
    [07/15/2024-22:53:29] [TRT] [I] Detected 15 inputs and 1 output network tensors.
    [07/15/2024-22:53:51] [TRT] [I] Total Host Persistent Memory: 111104
    [07/15/2024-22:53:51] [TRT] [I] Total Device Persistent Memory: 0
    [07/15/2024-22:53:51] [TRT] [I] Total Scratch Memory: 67125248
    [07/15/2024-22:53:51] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 626 steps to complete.
    [07/15/2024-22:53:51] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 37.0286ms to assign 17 blocks to 626 nodes requiring 229267456 bytes.
    [07/15/2024-22:53:51] [TRT] [I] Total Activation Memory: 229266432
    [07/15/2024-22:53:52] [TRT] [I] Total Weights Memory: 5340405760
    [07/15/2024-22:53:52] [TRT] [I] Engine generation completed in 25.4626 seconds.
    [07/15/2024-22:53:52] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 512 MiB, GPU 5093 MiB
    [07/15/2024-22:53:55] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 58606 MiB
    [07/15/2024-22:53:55] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:29
    [07/15/2024-22:53:57] [TRT-LLM] [I] Serializing engine to Llama-3-8B-Instruct-Gradient-1048k/trt_engines/rank2.engine...
    [07/15/2024-22:54:03] [TRT-LLM] [I] Engine serialized. Total time: 00:00:06
    [07/15/2024-22:55:04] [TRT-LLM] [I] Set dtype to float16.
    [07/15/2024-22:55:04] [TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 2317, GPU 809 (MiB)
    [07/15/2024-22:55:04] [TRT] [W] profileSharing0806 is on by default in TensorRT 10.0. This flag is deprecated and has no effect.
    [07/15/2024-22:55:04] [TRT-LLM] [I] Set nccl_plugin to float16.
    [07/15/2024-22:55:04] [TRT-LLM] [I] Set use_custom_all_reduce to True.
    [07/15/2024-22:55:04] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
    [07/15/2024-22:55:04] [TRT] [W] Unused Input: position_ids
    [07/15/2024-22:55:04] [TRT] [W] Detected layernorm nodes in FP16.
    [07/15/2024-22:55:04] [TRT] [W] Running layernorm after self-attention in FP16 may cause overflow. Exporting the model to the latest available ONNX opset (later than opset 17) to use the INormalizationLayer, or forcing layernorm layers to run in FP32 precision can help with preserving accuracy.
    [07/15/2024-22:55:04] [TRT] [W] [RemoveDeadLayers] Input Tensor position_ids is unused or used only at compile-time, but is not being removed.
    [07/15/2024-22:55:04] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
    [07/15/2024-22:55:07] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
    [07/15/2024-22:55:07] [TRT] [I] Detected 15 inputs and 1 output network tensors.
    [07/15/2024-22:55:31] [TRT] [I] Total Host Persistent Memory: 111104
    [07/15/2024-22:55:31] [TRT] [I] Total Device Persistent Memory: 0
    [07/15/2024-22:55:31] [TRT] [I] Total Scratch Memory: 67125248
    [07/15/2024-22:55:31] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 626 steps to complete.
    [07/15/2024-22:55:31] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 36.6399ms to assign 17 blocks to 626 nodes requiring 229267456 bytes.
    [07/15/2024-22:55:31] [TRT] [I] Total Activation Memory: 229266432
    [07/15/2024-22:55:31] [TRT] [I] Total Weights Memory: 5340405760
    [07/15/2024-22:55:31] [TRT] [I] Engine generation completed in 26.7251 seconds.
    [07/15/2024-22:55:31] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 512 MiB, GPU 5093 MiB
    [07/15/2024-22:55:34] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 58617 MiB
    [07/15/2024-22:55:35] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:30
    [07/15/2024-22:55:36] [TRT-LLM] [I] Serializing engine to Llama-3-8B-Instruct-Gradient-1048k/trt_engines/rank3.engine...
    [07/15/2024-22:55:42] [TRT-LLM] [I] Engine serialized. Total time: 00:00:06
    [07/15/2024-22:55:43] [TRT-LLM] [I] Total time of building all engines: 00:06:10
    CPU times: user 3.02 s, sys: 534 ms, total: 3.55 s
    Wall time: 6min 17s


  

  

  

# Prepare 1M needle-in-a-haystack datasets
### 8 seconds


```python
%%time 
# 8 seconds

!python ./TensorRT-LLM/examples/infinitebench/construct_synthetic_dataset.py \
    --test_case build_passkey \
    --test_level 7
```

### Inspect Synthetic Data


```python
!wc -c passkey.jsonl
!wc -w passkey.jsonl
!head -c 150 passkey.jsonl && printf '\n.....\n' && tail -c 250 passkey.jsonl
```

### Run Inference


```python
!mkdir -p 1M_context
```


```python
%%time
# <1 second

!mpirun -n 4   --allow-run-as-root python3 TensorRT-LLM/examples/eval_long_context.py \
               --task passkey \
               --engine_dir /home/rapids/notebooks/Llama-3-8B-Instruct-Gradient-1048k/trt_engines \
               --tokenizer_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
               --stop_idx 1 \
               --max_input_length 1048566 \
               --enable_chunked_context \
               --max_tokens_in_paged_kv_cache 1100000 \
               --output_dir ./1M_context
```

  
