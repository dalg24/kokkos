FROM nvidia/cuda:9.2-devel

RUN apt-get update && apt-get install -yq \
      wget bc pkg-config tree vim git cmake && \
    rm -rf /var/lib/apt/lists/*

# Install Clang/LLVM
ENV LLVM_DIR=/opt/llvm
RUN LLVM_VERSION=7.0.1 && \
    LLVM_KEY=86419D8A && \
    LLVM_URL=http://releases.llvm.org/${LLVM_VERSION}/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    LLVM_ARCHIVE=llvm-${LLVM_VERSION}.tar.xz && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${LLVM_URL} --output-document=${LLVM_ARCHIVE} && \
    wget --quiet ${LLVM_URL}.sig --output-document=${LLVM_ARCHIVE}.sig && \
    gpg --keyserver pool.sks-keyservers.net --recv-keys ${LLVM_KEY} && \
    gpg --verify ${LLVM_ARCHIVE}.sig ${LLVM_ARCHIVE} && \
    mkdir -p ${LLVM_DIR} && \
    tar -xvf ${LLVM_ARCHIVE} -C ${LLVM_DIR} --strip-components=1 && \
    echo "${LLVM_DIR}/lib" > /etc/ld.so.conf.d/llvm.conf && ldconfig && \
    rm -rf /root/.gnupg && \
    rm -rf ${SCRATCH_DIR}
ENV PATH=${LLVM_DIR}/bin:$PATH

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
