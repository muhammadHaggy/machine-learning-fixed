# Use NVIDIA CUDA devel image for compilation
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set basic environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# 3. Initialize Conda and Create Environment
COPY environment.yml /tmp/environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f /tmp/environment.yml

# IMPORTANT: Set the SHELL to use the new environment for all subsequent commands
SHELL ["conda", "run", "-n", "pointcept-torch2.5.0-cu12.4", "/bin/bash", "-c"]

# 4. Copy Source Code
WORKDIR /app
COPY . /app

# 5. Install Complex Dependencies & Server Deps
# Phase A: Custom Ops (No Build Isolation)
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation && \
    pip install ./libs/pointops --no-build-isolation && \
    pip install ./libs/pointgroup_ops --no-build-isolation

# Phase B: Runtime Deps & Server Deps
RUN pip install SharedArray scikit-learn flask requests gunicorn

# 6. Expose the port
EXPOSE 5001

# 7. Set Entrypoint
# We use conda run to execute gunicorn
# -w 1: One worker (Pointcept is heavy on GPU, don't run parallel workers on one GPU)
# -b 0.0.0.0:5001: Bind to all interfaces on port 5001
# --timeout 600: Allow 10 minutes for processing (adjust if large point clouds take longer)
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pointcept-torch2.5.0-cu12.4", "gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "--timeout", "600", "server:app"]