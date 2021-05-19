#pragma once

// HIP
#include <hip/hip_runtime_api.h>

// Google Benchmark
#include "benchmark/benchmark.h"

#include "cusparse_ops.hpp"

#include <type_traits>
#include <random>
#include <sstream>

using engine_type = std::ranlux48_base;
using seed_type = engine_type::result_type;

#define HIP_CHECK(condition)         \
{                                    \
    hipError_t _error = condition;    \
    if(_error != hipSuccess){         \
        std::cout << "HIP error: " << _error << " line: " << __LINE__ << std::endl; \
        exit(_error); \
    } \
}

namespace bench
{
    template<class T, class U, class V>
    inline auto get_random_data(size_t size, U min, V max, seed_type seed_value)
        -> typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
    {
        std::vector<T> data(size);
        std::generate(
            data.begin(),
            data.end(),
            [
                dist = std::uniform_real_distribution<T>{(T)min, (T)max},
                gen = engine_type{seed_value}
            ]() mutable
            {
                return static_cast<T>(dist(gen));
            }
        );
        return data;
    }

    template<class T, class U, class V>
    inline auto get_random_data(size_t size, U min, V max, seed_type seed_value)
        -> typename std::enable_if<std::is_same<T, cuDoubleComplex>::value || std::is_same<T, cuComplex>::value, std::vector<T>>::type
    {
        using real_type = decltype(cuAbs(cuGet<T>(0)));
        std::vector<T> data(size);
        std::generate(
            data.begin(),
            data.end(),
            [
                dist = std::uniform_real_distribution<real_type>{(real_type)min, (real_type)max},
                gen = engine_type{seed_value}
            ]() mutable -> T
            {
                return T{
                    static_cast<real_type>(dist(gen)),
                    static_cast<real_type>(dist(gen))
                };
            }
        );
        return data;
    }

    template <typename F>
    benchmark::internal::Benchmark* reg(const char* name, F&& f, int size, int sys_count)
    {
        std::stringstream ss;
        ss << name << "_" << size << "_" << sys_count;
        return benchmark::RegisterBenchmark(ss.str().c_str(), f, size, sys_count);
    }
}
