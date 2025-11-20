# ==========================================
# STAGE 1: Builder (Compiles everything)
# ==========================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

# 1. Install Build Dependencies
RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda
# We use the standard Miniconda installer
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh

# 3. Create Conda Environment (WITH TOS FIX)
COPY environment.yml /tmp/environment.yml

# We chain the ToS acceptance commands strictly before creation
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f /tmp/environment.yml

# Set Shell for compilation steps to use the new env
SHELL ["conda", "run", "-n", "pointcept-torch2.5.0-cu12.4", "/bin/bash", "-c"]

# 4. Compile Custom Ops
# Explicitly set GPU architectures to prevent "list index out of range" error
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

WORKDIR /app
COPY . /app

# Install deps (No build isolation allows seeing Conda PyTorch)
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation && \
    pip install ./libs/pointops --no-build-isolation && \
    pip install ./libs/pointgroup_ops --no-build-isolation && \
    pip install SharedArray scikit-learn flask requests gunicorn && \
    # CLEANUP: Remove caches to reduce the folder size we copy later
    conda clean -afy && \
    pip cache purge

# ==========================================
# STAGE 2: Runtime (The slim final image)
# ==========================================
# Switch to 'runtime' image (Saves ~3GB+ vs 'devel')
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Vital: We must use the exact same path for Conda as the builder stage
ENV PATH="/root/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin:$PATH"

# 1. Install ONLY Runtime Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the compiled Conda environment from the Builder stage
COPY --from=builder /root/miniconda3/envs/pointcept-torch2.5.0-cu12.4 /root/miniconda3/envs/pointcept-torch2.5.0-cu12.4

# 3. Copy Source Code
WORKDIR /app
COPY . /app

# 4. Setup Entrypoint
EXPOSE 5001

ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "--timeout", "600", "server:app"]