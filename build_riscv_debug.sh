#!/bin/bash

export CC=riscv64-unknown-linux-musl-gcc
export CXX=riscv64-unknown-linux-musl-g++

TPU_SDK_PATH=${TPU_SDK_PATH:-"cvitek_tpu_sdk"}
OPENCV_PATH=${TPU_SDK_PATH}/opencv

echo "Using TPU_SDK_PATH: $TPU_SDK_PATH"
echo "Using OPENCV_PATH: $OPENCV_PATH"

rm -rf build_riscv_debug
mkdir -p build_riscv_debug
cd build_riscv_debug

cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_C_FLAGS="-O2 -mcpu=c906fdv -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-O2 -mcpu=c906fdv -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" \
    -DCMAKE_CROSSCOMPILING=ON \
    -DTPU_SDK_PATH=${TPU_SDK_PATH} \
    -DOPENCV_PATH=${OPENCV_PATH} \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++"

make -j$(nproc)

echo "Build completed!"
echo "Executable: $(pwd)/tennis"
echo ""
echo "Usage: ./tennis <model.cvimodel> <input.jpg> [output.jpg]"
