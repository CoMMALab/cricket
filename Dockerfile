FROM mambaorg/micromamba:2.3.2

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY . cricket/
RUN cmake -GNinja -Bbuild -DCPM_SOURCE_CACHE=.cpm_cache cricket
RUN cmake --build build
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/tmp/build/fkcc_gen"]
