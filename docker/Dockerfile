FROM nvidia/cuda:9.2-devel

RUN apt-get update && apt-get install -yq \
      wget bc pkg-config tree vim git cmake && \
    rm -rf /var/lib/apt/lists/*

ARG KOKKOS_PREFIX
ARG KOKKOS_VERSION
ARG CONFIGURE_SCRIPT
COPY ${CONFIGURE_SCRIPT} /scratch/configure.sh

RUN KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    KOKKOS_ARCHIVE=kokkos.tar.gz && \
    cd /scratch && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p kokkos && \
    tar -xf ${KOKKOS_ARCHIVE} -C kokkos --strip-components=1 && \
    cd kokkos && export KOKKOS_PATH=$PWD && mkdir build && cd build && \
    ../../configure.sh && \
    make -j4 && make install && \
    rm -r /scratch
