FROM mambaorg/micromamba:jammy-cuda-12.2.2

ARG DEBIAN_FRONTEND=noninteractive

ARG NEW_MAMBA_USER=cgv
ARG NEW_MAMBA_USER_ID=1000
ARG NEW_MAMBA_USER_GID=1000
USER root

RUN usermod --login=${NEW_MAMBA_USER} --home=/home/${NEW_MAMBA_USER} \
            --move-home -u ${NEW_MAMBA_USER_ID} ${MAMBA_USER} \
    && groupmod --new-name=${NEW_MAMBA_USER} -g ${NEW_MAMBA_USER_GID} ${MAMBA_USER} \
    && echo "${NEW_MAMBA_USER}" > /etc/arg_mamba_user

ENV MAMBA_USER=$NEW_MAMBA_USER

ENV CUDA_HOME=/usr/local/cuda

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /tmp/env.yml

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV SHELL=/bin/bash

RUN --mount=type=cache,target=/opt/conda/pkgs,uid=${NEW_MAMBA_USER_ID},gid=${NEW_MAMBA_USER_GID} \
    --mount=type=cache,target=/home/${MAMBA_USER}/.mamba/pkgs,uid=${NEW_MAMBA_USER_ID},gid=${NEW_MAMBA_USER_GID} \
    --mount=type=cache,target=/home/${MAMBA_USER}/.cache/pip,uid=${NEW_MAMBA_USER_ID},gid=${NEW_MAMBA_USER_GID} \
    micromamba install -y -n base -f  /tmp/env.yml \
    && micromamba clean --all --yes \
    && rm /tmp/env.yml \
    && echo "micromamba activate base" >> ~/.bashrc

WORKDIR /home/$MAMBA_USER

CMD ["tail", "-f", "/dev/null"]
