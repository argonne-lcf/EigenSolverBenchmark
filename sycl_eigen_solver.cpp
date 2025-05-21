#include <complex>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

/* Auxiliary routine: printing a matrix */
void print_matrix(const char *desc, int m, int n, double *a, int lda)
{
    std::cout << "\n"
              << desc << "\n";
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            std::cout << " " << a[i + j * lda];
        std::cout << "\n";
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix size>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int lda = n;

    std::vector<double> A(lda * n);
    std::vector<double> w(n);

    /* Generate a random symmetric matrix */
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double value = dis(gen);
            A[i + j * lda] = value;
            A[j + i * lda] = value;
        }
    }

    /* Compute on device */
    sycl::queue device_queue(sycl::gpu_selector{});

    double *A_dev = sycl::malloc_device<double>(lda * n, device_queue);
    double *w_dev = sycl::malloc_device<double>(n, device_queue);

    int scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<double>(device_queue, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, n, lda);
    double *scratchpad_dev = sycl::malloc_device<double>(scratchpad_size, device_queue);

    device_queue.memcpy(A_dev, A.data(), sizeof(double) * lda * n);
    device_queue.wait();

    auto start = std::chrono::steady_clock::now();
    oneapi::mkl::lapack::syevd(device_queue, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, n, A_dev, lda, w_dev, scratchpad_dev, scratchpad_size);
    device_queue.wait();
    auto end = std::chrono::steady_clock::now();

    device_queue.memcpy(A.data(), A_dev, sizeof(double) * lda * n);
    device_queue.memcpy(w.data(), w_dev, sizeof(double) * n);
    device_queue.wait();

    /* Measure time */
    std::chrono::duration<double, std::milli> elapsed = end - start;

    /* Write to files */
    std::ofstream evals("sycl_evals_" + std::to_string(n) + ".txt");
    std::ofstream evecs("sycl_evecs_" + std::to_string(n) + ".txt");
    std::ofstream profile("sycl_profile.csv", std::ios::app);

    for (double eval : w)
        evals << eval << "\n";
    for (double evec : A)
        evecs << evec << "\n";
    profile << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << ", GPU, " << n << ", " << elapsed.count() << "\n";

    /* Clean up */
    sycl::free(A_dev, device_queue);
    sycl::free(w_dev, device_queue);
    sycl::free(scratchpad_dev, device_queue);

    return 0;
}