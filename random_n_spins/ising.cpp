#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

// Simulation parameters (defaults, can be overridden via CLI)
const double T_default = 2.0;    // Temperature (critical point is T_c = 2.269 for 2D Ising)
const double J = 1.0;            // Coupling constant
const int sweeps_default = 1000; // Number of Monte Carlo sweeps
const int seed_default = 123;    // RNG seed
const double k_B = 1.0;          // Boltzmann constant

int main(int argc, char *argv[])
{
    // Override defaults from CLI: <temperature> <sweeps> <seed> <save_mag> <save_snapshots>
    double T = T_default;
    int sweeps = sweeps_default;
    int seed = seed_default;
    bool save_magnetization = false;
    bool save_snapshots = false;

    if (argc >= 2)
        T = std::stod(argv[1]);
    if (argc >= 3)
        sweeps = std::stoi(argv[2]);
    if (argc >= 4)
        seed = std::stoi(argv[3]);
    if (argc >= 5)
        save_magnetization = (std::stoi(argv[4]) != 0);
    if (argc >= 6)
        save_snapshots = (std::stoi(argv[5]) != 0);

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

    std::cout << "Running with T=" << T
              << ", sweeps=" << sweeps
              << ", seed=" << seed
              << ", threads=" << num_threads
              << ", save_mag=" << save_magnetization
              << ", save_snapshots=" << save_snapshots << "\n";

    // Helper to dump the current grid to CSV
    auto write_snapshot = [&spins, n, N](const std::string &filename)
    {
        std::ofstream snapshot(filename);
        for (int j = 0; j < N; ++j)
        {
            snapshot << spins[j];
            if (j % n < n - 1)
                snapshot << ",";
            else
                snapshot << "\n";
        }
        snapshot.close();
    };

    std::ofstream mag_file;
    if (save_magnetization)
    {
        std::string mag_filename{"magnetization_" + std::to_string(seed) + ".txt"};
        mag_file.open(mag_filename);
        if (!mag_file.is_open())
        {
            std::cerr << "Error: cannot open " << mag_filename << '\n';
            return 1;
        }
    }

    if (save_snapshots)
    {
        write_snapshot("output_0.csv");
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

            // Only synchronize if we need to save data (avoids overhead when measuring performance)
            if (save_magnetization || save_snapshots)
            {
#pragma omp barrier

#pragma omp single
                {
                    int sweeps_done = sweep + 1;

                    if (save_magnetization)
                    {
                        // Calculate average magnetization
                        double mag_sum = 0.0;
                        for (int j = 0; j < N; ++j)
                        {
                            mag_sum += spins[j];
                        }
                        double avg_mag = mag_sum / N;
                        mag_file << sweep << " " << avg_mag << "\n";
                    }

                    if (save_snapshots && (sweeps_done % 1000 == 0 || sweeps_done == sweeps))
                    {
                        std::string filename = "output_" + std::to_string(sweeps_done) + ".csv";
                        write_snapshot(filename);
                    }
                }
            }
        }
    }

    if (save_magnetization)
    {
        mag_file.close();
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    std::cout << "Simulation finished (" << sweeps << " sweeps)\n";
    std::cout << "Time: " << elapsed_time << "s (" << num_threads << " threads)\n";

    bool file_exists = std::ifstream("metrics.csv").good();
    std::ofstream metrics("metrics.csv", std::ios::app);

    if (!file_exists)
    {
        metrics << "n_proc,sweeps,temperature,time\n";
    }

    metrics << num_threads << "," << sweeps << "," << T << "," << elapsed_time << "\n";
    metrics.close();

    std::ofstream output("output_last.csv");
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

    std::cout << "Result written to output_last.csv\n";

    return 0;
}