FROM mambaorg/micromamba:2.3.2 AS builder

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY . cricket/
RUN cmake -GNinja -Bbuild -DCPM_SOURCE_CACHE=.cpm_cache cricket && cmake --build build \
 && strip /tmp/build/fkcc_gen

FROM mambaorg/micromamba:2.3.2

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-runtime.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

COPY --from=builder /tmp/build/fkcc_gen /usr/local/bin/fkcc_gen
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/usr/local/bin/fkcc_gen"]
