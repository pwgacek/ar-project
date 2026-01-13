#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

// Simulation parameters
const double T = 2.0;   // Temperature (critical point is T_c = 2.269 for 2D Ising)
const double J = 1.0;   // Coupling constant
const int sweeps = 100; // Number of Monte Carlo sweeps
const int seed = 123;   // RNG seed
const double k_B = 1.0; // Boltzmann constant

int main()
{
    // Read input file
    std::ifstream input("input.csv");
    if (!input.is_open())
    {
        std::cerr << "Error: cannot open input.csv\n";
        return 1;
    }

    std::vector<int> spins;
    std::string line;

    while (std::getline(input, line))
    {
        // Parse CSV (spins separated by commas)
        size_t start = 0;
        while (start < line.length())
        {
            size_t end = line.find(',', start);
            if (end == std::string::npos)
                end = line.length();

            std::string token = line.substr(start, end - start);
            if (!token.empty())
            {
                spins.push_back(std::stoi(token));
            }
            start = end + 1;
        }
    }
    input.close();

    int N = spins.size();
    const int n{static_cast<int>(std::lround(std::sqrt(N)))};

    if (n * n != N)
    {
        std::cerr << "Error: N=" << N << " is not a perfect square\n";
        return 1;
    }

    std::cout << "Loaded " << N << " spins (" << n << "×" << n << " grid)\n";

    double beta = 1.0 / (k_B * T);

    int num_threads;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    double start_time = omp_get_wtime();

    std::mt19937 rng_master(seed);
    std::uniform_int_distribution<int> dist_spin(0, N - 1);

#pragma omp parallel // starts thread pool
    {
        // Variables private per thread
        std::mt19937 rng(seed + omp_get_thread_num());
        std::uniform_int_distribution<int> dist_site(0, N - 1);
        std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);

        for (int sweep = 0; sweep < sweeps; ++sweep) // every thread iterates over sweeps
        {
// N random steps (statistically attempts to flip every spin)
#pragma omp for schedule(guided)
            // N is divided across threads; e.g., 4 threads ideally compute ~N/4 steps each.
            for (int step = 0; step < N; ++step)
            {
                // Random spin candidate to flip
                int i = dist_site(rng);

                // 2D grid indices (row, col)
                int row = i / n;
                int col = i % n;

                // Periodic boundary conditions - 4 neighbors (up, down, left, right)
                int up = ((row - 1 + n) % n) * n + col;
                int down = ((row + 1) % n) * n + col;
                int left = row * n + (col - 1 + n) % n;
                int right = row * n + (col + 1) % n;

                // Energy before flip
                int S_i = spins[i];
                int neighbors_sum = spins[up] + spins[down] + spins[left] + spins[right];
                double E_old = -J * S_i * neighbors_sum;

                // Energy after flip
                int S_i_flipped = -S_i;
                double E_new = -J * S_i_flipped * neighbors_sum;

                // Energy change
                double dE = E_new - E_old;

                // Metropolis test
                // Note: potential race conditions are accepted as statistical noise
                if (dE < 0 || dist_uniform(rng) < std::exp(-beta * dE))
                {
                    // Metropolis criterion satisfied - update spin
                    spins[i] = S_i_flipped;
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    std::cout << "Simulation finished (" << sweeps << " sweeps)\n";
    std::cout << "Time: " << elapsed_time << "s (" << num_threads << " threads)\n";

    bool file_exists = std::ifstream("metrics.csv").good();
    std::ofstream metrics("metrics.csv", std::ios::app);

    if (!file_exists)
    {
        metrics << "num_threads,temperature,coupling_constant,sweeps,seed,boltzmann_constant,grid_size,time_seconds\n";
    }

    metrics << num_threads << "," << T << "," << J << "," << sweeps << ","
            << seed << "," << k_B << "," << n << "," << elapsed_time << "\n";
    metrics.close();

    std::ofstream output("output.csv");
    for (int i = 0; i < N; ++i)
    {
        output << spins[i];
        if (i % n < n - 1)
            output << ",";
        else
            output << "\n";
    }
    output << "\n";
    output.close();

    std::cout << "Result written to output.csv\n";

    return 0;
}