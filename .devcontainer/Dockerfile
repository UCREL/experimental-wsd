FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    git \
    make \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#ARG USERNAME=gpu
ARG USERNAME=ubuntu
#ARG USER_UID=1001
#ARG USER_GID=1000

# Create the user
#RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#    && usermod -s /bin/bash $USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
RUN ["/bin/bash", "-c", "set -o pipefail && wget -qO- https://astral.sh/uv/install.sh | sh"]
ENV PATH="/home/$USERNAME/.local/bin/:$PATH"
RUN ls -alh
RUN pwd
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-install-workspace
