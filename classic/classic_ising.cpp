#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>

// --- Simulation Parameters ---
const double J = 1.0;
const int SEED = 123;
const double k_B = 1.0;

struct Config {
    int L;
    int Px;
    int Py;
    int lx;
    int ly;
};

// Compute process grid dimensions
Config compute_process_grid(int L, int size) {
    Config cfg;
    cfg.L = L;
    
    // Find Px and Py such that Px * Py = size
    cfg.Px = static_cast<int>(std::sqrt(size));
    while (size % cfg.Px != 0) {
        cfg.Px--;
    }
    cfg.Py = size / cfg.Px;
    
    if (cfg.Px * cfg.Py != size) {
        throw std::runtime_error("Could not factor worker size into a Px x Py grid");
    }
    if (L % cfg.Px != 0 || L % cfg.Py != 0) {
        throw std::runtime_error("Grid size L must be divisible by Px and Py");
    }
    
    cfg.lx = L / cfg.Px;
    cfg.ly = L / cfg.Py;
    
    return cfg;
}

// Load initial grid from CSV file (rank 0 only)
std::vector<int8_t> load_initial_grid(const std::string& path, int& L) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
    
    std::vector<std::vector<int>> temp_grid;
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stoi(value));
        }
        
        if (!row.empty()) {
            temp_grid.push_back(row);
        }
    }
    
    file.close();
    
    if (temp_grid.empty() || temp_grid.size() != temp_grid[0].size()) {
        throw std::runtime_error("Input grid must be square 2D");
    }
    
    L = temp_grid.size();
    std::vector<int8_t> grid(L * L);
    
    // Flatten grid (input is always -1/+1 format)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i * L + j] = static_cast<int8_t>(temp_grid[i][j]);
        }
    }
    
    return grid;
}

// Save grid to CSV file
void save_grid(const std::string& path, const std::vector<int8_t>& grid, int L) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + path);
    }
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            file << static_cast<int>(grid[i * L + j]);
            if (j < L - 1) file << ",";
        }
        file << "\n";
    }
    
    file.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    int SWEEPS;
    double T;
    bool save_magnetization = false;
    bool use_random_seed = false;
    bool save_output = false;
    
    if (argc >= 3) {
        SWEEPS = std::atoi(argv[1]);
        T = std::atof(argv[2]);
        
        // Check for flags
        for (int i = 3; i < argc; i++) {
            if (std::strcmp(argv[i], "--save-magnetization") == 0) {
                save_magnetization = true;
            } else if (std::strcmp(argv[i], "--random-seed") == 0) {
                use_random_seed = true;
            } else if (std::strcmp(argv[i], "--save-output") == 0) {
                save_output = true;
            }
        }
    } else if (rank == 0) {
        std::cerr << "Usage: " << argv[0] << " <sweeps> <temperature> [--save-magnetization] [--random-seed] [--save-output]" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // --- Step 1: Configuration & Data Distribution ---
    Config cfg;
    std::vector<int8_t> global_grid;
    
    if (rank == 0) {
        // Load data only on Master
        int L;
        global_grid = load_initial_grid("input.csv", L);
        cfg = compute_process_grid(L, size);
    }
    
    // Broadcast configuration
    MPI_Bcast(&cfg, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Prepare local block buffer
    std::vector<int8_t> initial_block(cfg.ly * cfg.lx);
    
    if (rank == 0) {
        // Send blocks to other ranks
        for (int r = 0; r < size; r++) {
            int rpx = r % cfg.Px;
            int rpy = r / cfg.Px;
            int x0 = rpx * cfg.lx;
            int y0 = rpy * cfg.ly;
            
            std::vector<int8_t> block_to_send(cfg.ly * cfg.lx);
            for (int i = 0; i < cfg.ly; i++) {
                for (int j = 0; j < cfg.lx; j++) {
                    block_to_send[i * cfg.lx + j] = global_grid[(y0 + i) * cfg.L + (x0 + j)];
                }
            }
            
            if (r == 0) {
                initial_block = block_to_send;
            } else {
                MPI_Send(block_to_send.data(), cfg.ly * cfg.lx, MPI_INT8_T, r, 77, MPI_COMM_WORLD);
            }
        }
    } else {
        // Workers receive their block
        MPI_Recv(initial_block.data(), cfg.ly * cfg.lx, MPI_INT8_T, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // --- Step 2: Simulation Setup ---
    
    // Determine seed based on flag (only on rank 0, then broadcast)
    int base_seed;
    if (rank == 0) {
        if (use_random_seed) {
            // Use current time as seed
            base_seed = static_cast<int>(std::chrono::high_resolution_clock::now().time_since_epoch().count() & 0xFFFFFFFF);
            std::cout << "Using random seed: " << base_seed << std::endl;
        } else {
            base_seed = SEED;
        }
    }
    
    // Broadcast seed to all ranks to ensure synchronization
    MPI_Bcast(&base_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    double beta = 1.0 / (k_B * T);
    std::mt19937_64 per_rank_rng(base_seed + 1000003 * rank);
    std::mt19937_64 common_rng(base_seed);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::uniform_int_distribution<int> row_dist(1, cfg.ly);
    std::uniform_int_distribution<int> col_dist(1, cfg.lx);
    
    // Find this process's coordinates
    int px = rank % cfg.Px;
    int py = rank / cfg.Px;
    
    // Local spins with ghost borders
    std::vector<int8_t> spins((cfg.ly + 2) * (cfg.lx + 2), 0);  // Initialize all to 0
    for (int i = 0; i < cfg.ly; i++) {
        for (int j = 0; j < cfg.lx; j++) {
            spins[(i + 1) * (cfg.lx + 2) + (j + 1)] = initial_block[i * cfg.lx + j];
        }
    }
    
    // Neighbor ranks
    int up = ((py - 1 + cfg.Py) % cfg.Py) * cfg.Px + px;
    int down = ((py + 1) % cfg.Py) * cfg.Px + px;
    int left = py * cfg.Px + ((px - 1 + cfg.Px) % cfg.Px);
    int right = py * cfg.Px + ((px + 1) % cfg.Px);
    
    // Preallocate buffers
    std::vector<int8_t> send_left(cfg.ly);
    std::vector<int8_t> send_right(cfg.ly);
    std::vector<int8_t> recv_left(cfg.ly);
    std::vector<int8_t> recv_right(cfg.ly);
    
    // Exchange ghost cells
    auto exchange_ghosts = [&]() {
        const int TAG_ROW_UP = 10;
        const int TAG_ROW_DOWN = 11;
        const int TAG_COL_LEFT = 20;
        const int TAG_COL_RIGHT = 21;
        
        // Rows
        MPI_Sendrecv(&spins[1 * (cfg.lx + 2) + 1], cfg.lx, MPI_INT8_T, up, TAG_ROW_UP,
                     &spins[(cfg.ly + 1) * (cfg.lx + 2) + 1], cfg.lx, MPI_INT8_T, down, TAG_ROW_UP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&spins[cfg.ly * (cfg.lx + 2) + 1], cfg.lx, MPI_INT8_T, down, TAG_ROW_DOWN,
                     &spins[0 * (cfg.lx + 2) + 1], cfg.lx, MPI_INT8_T, up, TAG_ROW_DOWN,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Columns
        for (int i = 0; i < cfg.ly; i++) {
            send_left[i] = spins[(i + 1) * (cfg.lx + 2) + 1];
            send_right[i] = spins[(i + 1) * (cfg.lx + 2) + cfg.lx];
        }
        
        MPI_Sendrecv(send_left.data(), cfg.ly, MPI_INT8_T, left, TAG_COL_LEFT,
                     recv_right.data(), cfg.ly, MPI_INT8_T, right, TAG_COL_LEFT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(send_right.data(), cfg.ly, MPI_INT8_T, right, TAG_COL_RIGHT,
                     recv_left.data(), cfg.ly, MPI_INT8_T, left, TAG_COL_RIGHT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for (int i = 0; i < cfg.ly; i++) {
            spins[(i + 1) * (cfg.lx + 2) + 0] = recv_left[i];
            spins[(i + 1) * (cfg.lx + 2) + cfg.lx + 1] = recv_right[i];
        }
    };
    
    // Metropolis lookup table
    double exp_4J = std::exp(-beta * 4 * J);
    double exp_8J = std::exp(-beta * 8 * J);
    
    // Staleness tracking
    std::vector<bool> stale_top(cfg.lx, false);
    std::vector<bool> stale_bottom(cfg.lx, false);
    std::vector<bool> stale_left(cfg.ly, false);
    std::vector<bool> stale_right(cfg.ly, false);
    
    exchange_ghosts();
    
    // --- Step 3: Simulation Loop ---
    std::ofstream* out = nullptr;
    std::string output_dir;
    if (rank == 0) {
        if (save_output) {
            // Create output directory with temperature in name
            std::stringstream dir_name;
            dir_name << "output_" << std::fixed << std::setprecision(2) << T;
            output_dir = dir_name.str();
            mkdir(output_dir.c_str(), 0755);
            
            // Save initial state (sweep 0)
            std::vector<int8_t> initial_local_copy(cfg.ly * cfg.lx);
            for (int i = 0; i < cfg.ly; i++) {
                for (int j = 0; j < cfg.lx; j++) {
                    initial_local_copy[i * cfg.lx + j] = spins[(i + 1) * (cfg.lx + 2) + (j + 1)];
                }
            }
            
            std::vector<int8_t> initial_gathered(size * cfg.ly * cfg.lx);
            MPI_Gather(initial_local_copy.data(), cfg.ly * cfg.lx, MPI_INT8_T,
                       initial_gathered.data(), cfg.ly * cfg.lx, MPI_INT8_T,
                       0, MPI_COMM_WORLD);
            
            std::vector<int8_t> initial_global(cfg.L * cfg.L);
            for (int r = 0; r < size; r++) {
                int rpx = r % cfg.Px;
                int rpy = r / cfg.Px;
                int x0 = rpx * cfg.lx;
                int y0 = rpy * cfg.ly;
                
                for (int i = 0; i < cfg.ly; i++) {
                    for (int j = 0; j < cfg.lx; j++) {
                        initial_global[(y0 + i) * cfg.L + (x0 + j)] = initial_gathered[r * cfg.ly * cfg.lx + i * cfg.lx + j];
                    }
                }
            }
            
            std::stringstream initial_filename;
            initial_filename << output_dir << "/output_0.csv";
            save_grid(initial_filename.str(), initial_global, cfg.L);
        } else {
            // If not saving output, just participate in gather for rank 0
            std::vector<int8_t> initial_local_copy(cfg.ly * cfg.lx);
            for (int i = 0; i < cfg.ly; i++) {
                for (int j = 0; j < cfg.lx; j++) {
                    initial_local_copy[i * cfg.lx + j] = spins[(i + 1) * (cfg.lx + 2) + (j + 1)];
                }
            }
            
            std::vector<int8_t> initial_gathered(size * cfg.ly * cfg.lx);
            MPI_Gather(initial_local_copy.data(), cfg.ly * cfg.lx, MPI_INT8_T,
                       initial_gathered.data(), cfg.ly * cfg.lx, MPI_INT8_T,
                       0, MPI_COMM_WORLD);
        }
    } else {
        // Workers participate in gather
        std::vector<int8_t> initial_local_copy(cfg.ly * cfg.lx);
        for (int i = 0; i < cfg.ly; i++) {
            for (int j = 0; j < cfg.lx; j++) {
                initial_local_copy[i * cfg.lx + j] = spins[(i + 1) * (cfg.lx + 2) + (j + 1)];
            }
        }
        
        MPI_Gather(initial_local_copy.data(), cfg.ly * cfg.lx, MPI_INT8_T,
                   nullptr, 0, MPI_INT8_T,
                   0, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        if (save_output) {
            // Create output directory with temperature in name
            std::stringstream dir_name;
            dir_name << "output_" << std::fixed << std::setprecision(2) << T;
            output_dir = dir_name.str();
            mkdir(output_dir.c_str(), 0755);
        }
        
        if (save_magnetization) {
            // Create magnetization directory with temperature in name
            std::stringstream mag_dir_name;
            mag_dir_name << "magnetization_" << std::fixed << std::setprecision(2) << T;
            std::string mag_dir = mag_dir_name.str();
            mkdir(mag_dir.c_str(), 0755);
            
            // Generate filename with timestamp
            auto now = std::chrono::system_clock::now();
            auto now_c = std::chrono::system_clock::to_time_t(now);
            std::stringstream filename;
            filename << mag_dir << "/magnetization_" 
                     << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S")
                     << ".txt";
            
            out = new std::ofstream(filename.str());
        }
    }
    
    for (int sweep = 0; sweep < SWEEPS; sweep++) {
        for (int step = 0; step < cfg.ly * cfg.lx; step++) {
            int i = row_dist(common_rng);
            int j = col_dist(common_rng);
            
            bool is_border = (i == 1) || (i == cfg.ly) || (j == 1) || (j == cfg.lx);
            
            if (is_border) {
                bool local_need_sync = false;
                
                if (i == 1 && stale_top[j - 1]) local_need_sync = true;
                if (i == cfg.ly && stale_bottom[j - 1]) local_need_sync = true;
                if (j == 1 && stale_left[i - 1]) local_need_sync = true;
                if (j == cfg.lx && stale_right[i - 1]) local_need_sync = true;
                
                bool global_need_sync = false;
                MPI_Allreduce(&local_need_sync, &global_need_sync, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
                
                if (global_need_sync) {
                    exchange_ghosts();
                    std::fill(stale_top.begin(), stale_top.end(), false);
                    std::fill(stale_bottom.begin(), stale_bottom.end(), false);
                    std::fill(stale_left.begin(), stale_left.end(), false);
                    std::fill(stale_right.begin(), stale_right.end(), false);
                }
            }
            
            int idx = i * (cfg.lx + 2) + j;
            int8_t s = spins[idx];
            int nn = spins[(i+1) * (cfg.lx + 2) + j] + 
                     spins[(i-1) * (cfg.lx + 2) + j] + 
                     spins[i * (cfg.lx + 2) + (j+1)] + 
                     spins[i * (cfg.lx + 2) + (j-1)];
            
            int dE = 2 * J * s * nn;
            
            bool flip = false;
            if (dE <= 0) {
                flip = true;
            } else {
                double p;
                if (dE == 4 * J) {
                    p = exp_4J;
                } else if (dE == 8 * J) {
                    p = exp_8J;
                } else {
                    p = std::exp(-beta * dE);
                }
                
                if (uniform_dist(per_rank_rng) < p) {
                    flip = true;
                }
            }
            
            if (flip) {
                spins[idx] = -s;
            }
            
            if (is_border) {
                if (j == 1) stale_left[i - 1] = true;
                if (j == cfg.lx) stale_right[i - 1] = true;
                if (i == 1) stale_top[j - 1] = true;
                if (i == cfg.ly) stale_bottom[j - 1] = true;
            }
        }
        
        // Calculate Magnetization
        int64_t local_M = 0;
        for (int i = 1; i <= cfg.ly; i++) {
            for (int j = 1; j <= cfg.lx; j++) {
                local_M += spins[i * (cfg.lx + 2) + j];
            }
        }
        
        int64_t M = 0;
        MPI_Reduce(&local_M, &M, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && save_magnetization) {
            double mag_per_spin = static_cast<double>(M) / (cfg.L * cfg.L);
            *out << sweep << " " << std::fixed << mag_per_spin << "\n";
        }
        
        // Save grid every 1000 sweeps if enabled
        if (save_output && (sweep + 1) % 1000 == 0) {
            // Gather current state
            std::vector<int8_t> current_local(cfg.ly * cfg.lx);
            for (int i = 0; i < cfg.ly; i++) {
                for (int j = 0; j < cfg.lx; j++) {
                    current_local[i * cfg.lx + j] = spins[(i + 1) * (cfg.lx + 2) + (j + 1)];
                }
            }
            
            std::vector<int8_t> current_gathered;
            if (rank == 0) {
                current_gathered.resize(size * cfg.ly * cfg.lx);
            }
            
            MPI_Gather(current_local.data(), cfg.ly * cfg.lx, MPI_INT8_T,
                       current_gathered.data(), cfg.ly * cfg.lx, MPI_INT8_T,
                       0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                std::vector<int8_t> current_global(cfg.L * cfg.L);
                for (int r = 0; r < size; r++) {
                    int rpx = r % cfg.Px;
                    int rpy = r / cfg.Px;
                    int x0 = rpx * cfg.lx;
                    int y0 = rpy * cfg.ly;
                    
                    for (int i = 0; i < cfg.ly; i++) {
                        for (int j = 0; j < cfg.lx; j++) {
                            current_global[(y0 + i) * cfg.L + (x0 + j)] = current_gathered[r * cfg.ly * cfg.lx + i * cfg.lx + j];
                        }
                    }
                }
                
                std::stringstream output_filename;
                output_filename << output_dir << "/output_" << (sweep + 1) << ".csv";
                save_grid(output_filename.str(), current_global, cfg.L);
            }
        }
    }
    
    if (rank == 0 && save_magnetization) {
        out->close();
        delete out;
    }
    
    // --- Step 4: Finish ---
    if (rank == 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "Simulation completed in " << std::fixed << elapsed.count() << " seconds" << std::endl;
        
        std::ofstream results("results.csv", std::ios::app);
        results << size << "," << SWEEPS << "," << T << "," << std::fixed << elapsed.count() << "\n";
        results.close();
    }
    
    MPI_Finalize();
    return 0;
}
