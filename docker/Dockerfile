# GraphVSelect — inference environment.
# The method is entirely training-free: every component is a frozen off-the-shelf
# model served for inference only (no optimizer, no fine-tuning).
#
# Base image ships PyTorch 2.5 / CUDA 12.1; installing vLLM 0.11 upgrades the
# runtime to torch 2.8.0 (CUDA 12.8) wheels, which is what the paper reports.
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# vLLM 0.11 is the first release with native Qwen3-VL multi-modal support.
# Pinned exactly: 0.20+ requires CUDA combinations beyond driver 575 (CUDA 12.9).
RUN pip install --upgrade pip && \
    pip install "vllm==0.11.0"

# transformers 5.x removed a slow-tokenizer attribute that vLLM 0.11 still relies
# on; stay on the last 4.x line (recognises qwen3_vl, keeps the attribute).
RUN pip install "transformers>=4.55,<5.0"

# Qwen3-VL helper utilities (per the vLLM recipe).
RUN pip install "qwen-vl-utils==0.0.14"

# vLLM 0.11 imports ImageChunk from the pre-1.9 mistral_common layout.
RUN pip install "mistral_common[audio,image]>=1.8.2,<1.9"

# Project-level dependencies.
RUN pip install \
    loguru \
    PyYAML \
    Pillow \
    pandas \
    pyarrow \
    pycocotools \
    opencv-python-headless \
    matplotlib \
    accelerate \
    spacy \
    nltk

# spaCy English model — noun-phrase extraction from the query.
RUN python -m spacy download en_core_web_sm

# NLTK WordNet — fallback when the detector returns no boxes.
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# HF_HOME is intentionally left unset here; mount a host cache at run time, e.g.
#   docker run --gpus all -v $PWD:/workspace/GraphVSelect -v <hf_cache>:/hf \
#              -e HF_HOME=/hf graphvselect:latest ...
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/GraphVSelect \
    PYTHONHASHSEED=0

WORKDIR /workspace/GraphVSelect
