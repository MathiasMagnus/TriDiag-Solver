#pragma once

// HIP
#include <hip/hip_runtime_api.h>

// Google Test
#include <gtest/gtest.h>

#include "cusparse_ops.hpp"

#include <type_traits>
#include <random>

using engine_type = std::ranlux48_base;
using seed_type = engine_type::result_type;

#include <hip/hip_runtime.h>

#define HIP_CHECK(condition)         \
{                                    \
    hipError_t _error = condition;    \
    if(_error != hipSuccess){         \
        std::cout << "HIP error: " << _error << " line: " << __LINE__ << std::endl; \
        exit(_error); \
    } \
}

namespace test
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

    template<class T>
    void tridiag_mul(std::vector<T>& x, const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c, const std::vector<T>& d)
    {
        auto m = 1*b.size();
        auto rhs = d.size()/b.size();
        for(decltype(rhs) j=0; j<rhs; j++)
        {
            x[j*m] =  cuAdd( cuMul(b[0],d[j*m]), cuMul(c[0],d[j*m+1]));
            for(int i=1; i<m-1; i++)
            {
                //x[i]=  a[i]*d[i-1]+b[i]*d[i]+c[i]*d[i+1];
                x[j*m+i]=  cuMul(a[i],d[j*m+i-1]);
                x[j*m+i]=  cuFma(b[i], d[j*m+i], x[j*m+i]);
                x[j*m+i]=  cuFma(c[i], d[j*m+i+1], x[j*m+i]);
            }
            x[j*m+m-1]= cuAdd( cuMul(a[m-1],d[j*m+m-2]) , cuMul(b[m-1],d[j*m+m-1]) );
        }
    }

    template<class T, class U>
    auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const U percent)
        -> typename std::enable_if<std::is_floating_point<T>::value>::type
    {
        ASSERT_EQ(result.size(), expected.size());
        for(size_t i = 0; i < result.size(); i++)
        {
            auto diff = std::max<U>(std::abs(percent * expected[i]), percent);
            ASSERT_NEAR(result[i], expected[i], diff) << "where index = " << i;
        }
    }

    template<class T, class U>
    auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const U percent)
        -> typename std::enable_if<std::is_same<T, cuDoubleComplex>::value || std::is_same<T, cuComplex>::value>::type
    {
        //ASSERT_EQ(result.size(), expected.size());
        for(size_t i = 0; i < result.size(); i++)
        {
            auto diff = std::max<U>(cuAbs(cuMul(cuGet<T>(percent), expected[i])), percent);
            ASSERT_NEAR(cuReal(result[i]), cuReal(expected[i]), diff) << "where index = " << i;
            ASSERT_NEAR(cuImag(result[i]), cuImag(expected[i]), diff) << "where index = " << i;
        }
    }
}
