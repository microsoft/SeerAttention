/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

inline int get_current_device() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get current CUDA device");
    }
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
    cudaError_t err1 = cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaError_t err2 = cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device);
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        throw std::runtime_error("Failed to get compute capability");
    }
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    cudaError_t err = cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get number of SMs");
    }
    return multiprocessor_count;
}
