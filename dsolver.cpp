#include <complex>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// Define to enable writing eigenvalues and eigenvectors to files
// #define WRITE_RESULTS

/* Auxiliary routine: printing a matrix */
void print_matrix(const char *desc, MKL_INT64 m, MKL_INT64 n, double *a, MKL_INT64 lda)
{
    std::cout << "\n"
              << desc << "\n";
    for (MKL_INT64 i = 0; i < m; i++)
    {
        for (MKL_INT64 j = 0; j < n; j++)
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

    MKL_INT64 n = std::stoll(argv[1]);
    MKL_INT64 lda = n;

    try
    {
        std::vector<double> A(lda * n);
        std::vector<double> w(n);

        /* Generate a random symmetric matrix */
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> dis(-10.0, 10.0);

        for (MKL_INT64 i = 0; i < n; i++)
        {
            for (MKL_INT64 j = 0; j <= i; j++)
            {
                double value = dis(gen);
                A[i + j * lda] = value;
                A[j + i * lda] = value;
            }
        }

        /* Compute on device */
        sycl::queue device_queue(sycl::gpu_selector_v);

        double *A_dev = sycl::malloc_device<double>(lda * n, device_queue);
        double *w_dev = sycl::malloc_device<double>(n, device_queue);

        // Calculate scratchpad size with error checking
        MKL_INT64 scratchpad_size = 0;
        try
        {
            scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<double>(
                device_queue, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, n, lda);
        }
        catch (const oneapi::mkl::lapack::invalid_argument &e)
        {
            std::cerr << "Error calculating scratchpad size: " << e.what() << std::endl;
            std::cerr << "Matrix size: " << n << "x" << n << std::endl;
            return 1;
        }

        if (scratchpad_size <= 0)
        {
            std::cerr << "Invalid scratchpad size: " << scratchpad_size << std::endl;
            return 1;
        }

        double *scratchpad_dev = sycl::malloc_device<double>(scratchpad_size, device_queue);

        device_queue.memcpy(A_dev, A.data(), sizeof(double) * lda * n);
        device_queue.wait();

        auto start = std::chrono::steady_clock::now();
        try
        {
            oneapi::mkl::lapack::syevd(device_queue, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper,
                                       n, A_dev, lda, w_dev, scratchpad_dev, scratchpad_size);
        }
        catch (const oneapi::mkl::lapack::invalid_argument &e)
        {
            std::cerr << "Error in syevd: " << e.what() << std::endl;
            std::cerr << "Matrix size: " << n << "x" << n << std::endl;
            std::cerr << "Scratchpad size: " << scratchpad_size << std::endl;
            return 1;
        }
        device_queue.wait();
        auto end = std::chrono::steady_clock::now();

        device_queue.memcpy(A.data(), A_dev, sizeof(double) * lda * n);
        device_queue.memcpy(w.data(), w_dev, sizeof(double) * n);
        device_queue.wait();

        /* Measure time */
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Computation time: " << elapsed.count() << " ms" << std::endl;

#ifdef WRITE_RESULTS
        /* Write to files */
        std::ofstream evals("sycl_evals_" + std::to_string(n) + ".txt");
        std::ofstream evecs("sycl_evecs_" + std::to_string(n) + ".txt");
        std::ofstream profile("sycl_profile.csv", std::ios::app);

        for (double eval : w)
            evals << eval << "\n";
        for (double evec : A)
            evecs << evec << "\n";
        profile << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << ", GPU, " << n << ", " << elapsed.count() << "\n";
#endif

        /* Clean up */
        sycl::free(A_dev, device_queue);
        sycl::free(w_dev, device_queue);
        sycl::free(scratchpad_dev, device_queue);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}: