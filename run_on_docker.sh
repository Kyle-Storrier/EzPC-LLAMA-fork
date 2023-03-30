#!/bin/bash

SRC_NAME="EzPC-LLAMA-fork"
SRC_DIR="/home/kyle/$SRC_NAME"
TARGET_CONTAINER="llama-fork-test"
TEST_MICROBENCHMARK="tanh"

docker cp $SRC_DIR $TARGET_CONTAINER:/ezpc_dir/

docker exec -w /ezpc_dir/$SRC_NAME/FSS $TARGET_CONTAINER bash -c "mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Release ../ && make install"

docker exec -w /ezpc_dir/EzPC-LLAMA-fork/FSS/microbenchmarks $TARGET_CONTAINER g++ -Wno-ignored-attributes -O3 -pthread -std=c++17 -I/ezpc_dir/EzPC-LLAMA-fork/EzPC/EzPC/../../FSS/build/install/include ./$TEST_MICROBENCHMARK.cpp /ezpc_dir/EzPC-LLAMA-fork/EzPC/EzPC/../../FSS/build/install/lib/libfss.a -o ./$TEST_MICROBENCHMARK.out
# docker exec -w /ezpc_dir/EzPC/FSS/microbenchmarks/ $TARGET_CONTAINER fssc --bitlen 64 $TEST_MICROBENCHMARK.ezpc

