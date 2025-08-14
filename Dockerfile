FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -qqy lsb-release curl

RUN mkdir -p /etc/apt/keyrings && \
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
    | tee /etc/apt/keyrings/robotpkg.asc

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list

RUN apt-get update && \
    apt-get install -qqy \
      robotpkg-py3*-pinocchio \
      libcppad-dev \
      libeigen3-dev \
      libcgal-dev \
      git \
      patch \
      cmake \
      ninja-build \
      nlohmann-json3-dev \
      build-essential \
      libfmt-dev \
      libboost-all-dev

ENV PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
COPY . cricket/
RUN cmake -GNinja -Bbuild cricket && cmake --build build

ENTRYPOINT ["/build/fkcc_gen"]
