FROM mambaorg/micromamba:2.3.0
ENV DEBIAN_FRONTEND=noninteractive

RUN micromamba create -n cricket -c conda-forge pinocchio cppad eigen cgal inja nlohmann_json cmake cxx-compiler llvm ninja fmt pkg-config
SHELL ["micromamba", "run", "-n", "cricket", "/bin/bash", "-c"]

RUN mkdir -p cricket
COPY . cricket/
RUN cmake -GNinja -Bcricket/build cricket && cmake --build cricket/build
