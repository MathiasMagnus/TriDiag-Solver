#include "tridiag_test_hip.hpp"

#include "tridiag_solver.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <vector>
#include <utility>
#include <random>
#include <cmath>
#include <complex>

template<class T>
struct params
{
    using storage_type = T;
    // TODO: internal_type comes here when tridiag_solver
    //       learns how to calculate in a different type
    //       than the type of input/output
};

using types = ::testing::Types<
    params<float>, // TODO: float seems to have terrible accuracy
    params<double>,
    params<cuComplex>,
    params<cuDoubleComplex>
>;

std::initializer_list<int> system_counts = { 1, 2, 3, 4 };
std::initializer_list<int> system_sizes = {
    127, 128, 129,
    255, 256, 257,
    500, 768, 1000,
    1023, 1024, 1025,
    1234,
    /*2047,*/ 2048, 2049, // TODO: 2047 is buggy
    4096, 4097,
    32768
};
std::initializer_list<seed_type> seeds = { 500 };
template<class T> constexpr T abs_threshold;
template<> const float abs_threshold<float> = (float)1e-4;
template<> const double abs_threshold<double> = (double)1e-6;
template<> const float abs_threshold<cuComplex> = abs_threshold<float> * std::sqrt(2.f);
template<> const double abs_threshold<cuDoubleComplex> = abs_threshold<double> * std::sqrt(2.);

template<class Params>
class solve_tests : public ::testing::Test
{
public:
    using storage_type = typename Params::storage_type;
    // TODO: add internal_type here
};

// Deduplicates common test logic
template <typename T, typename F>
void for_each_config(F&& f)
{
    int device_id;
    hipDeviceProp_t device_props;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&device_props, device_id));
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id << " (" << device_props.name << ")");
    HIP_CHECK(hipSetDevice(device_id));

    for (seed_type seed : seeds)
    {
        SCOPED_TRACE(testing::Message() << "with seed= " << seed);
        for (auto size : system_sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for (auto sys_count : system_counts)
            {
                SCOPED_TRACE(testing::Message() << "with systems = " << sys_count);
                f(seed, size, sys_count);
            }
        }
    }
}

template <typename T>
void process_and_test(const std::vector<T>& dl, const std::vector<T>& d, const std::vector<T>& du, const std::vector<T>& b)
{
    int size = (int)d.size(),
        sys_count = (int)(b.size()/d.size());
    T* dev_dl; HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_dl), dl.size() * sizeof(T)));
    T* dev_d;  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_d ), d .size() * sizeof(T)));
    T* dev_du; HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_du), du.size() * sizeof(T)));
    T* dev_b;  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_b ), b .size() * sizeof(T)));

    HIP_CHECK(hipMemcpy(dev_dl, dl.data(), dl.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_d , d .data(), d .size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_du, du.data(), du.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b,  b .data(), b .size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    tridiag_solver<T> tds{size, sys_count};
    tds.solve(dev_dl, dev_d, dev_du, dev_b);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    std::vector<T> x(size*sys_count);
    HIP_CHECK(hipMemcpy(x.data(), dev_b, x.size() * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Apply solution
    std::vector<T> b_new(size*sys_count);
    test::tridiag_mul(b_new, dl, d, du, x);

    // Check if output values are as expected
    test::assert_near(b_new, b, abs_threshold<T>);

    HIP_CHECK(hipFree(dev_dl));
    HIP_CHECK(hipFree(dev_d));
    HIP_CHECK(hipFree(dev_du));
    HIP_CHECK(hipFree(dev_b));
}

TYPED_TEST_SUITE(solve_tests, types);

TYPED_TEST(solve_tests, identity)
{
    using T = typename TestFixture::storage_type;
    // TODO: add internal type here

    for_each_config<T>([](seed_type seed, int size, int sys_count)
    {
        // Generate data
        std::vector<T> dl(size, cuGet<T>(0)),
                       d (size, cuGet<T>(1)),
                       du(size, cuGet<T>(0)),
                       b  = test::get_random_data<T>(size*sys_count, 0, 32768, seed++);

        process_and_test<T>(dl, d, du, b);
    });
}

TYPED_TEST(solve_tests, fully_random)
{
    using T = typename TestFixture::storage_type;
    // TODO: add internal type here

    for_each_config<T>([](seed_type seed, int size, int sys_count)
    {
        // Generate data
        std::vector<T> dl = test::get_random_data<T>(size, 0, 32768, seed++),
                       d  = test::get_random_data<T>(size, 0, 32768, seed++),
                       du = test::get_random_data<T>(size, 0, 32768, seed++),
                       b  = test::get_random_data<T>(size*sys_count, 0, 32768, seed++);
        dl[0] = cuGet<T>(0);
        du[size-1] = cuGet<T>(0);

        process_and_test<T>(dl, d, du, b);
    });
}
