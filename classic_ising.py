#!/usr/bin/env python

import time
import numpy as np
from mpi4py import MPI

# --- Simulation Parameters ---
T = 2.5
J = 1.0
sweeps = 20
seed = 123
k_B = 1.0

def _compute_process_grid(L: int, size: int) -> tuple[int, int, int, int]:
    """Computes the Px * Py grid dimensions."""
    Px = int(np.sqrt(size))
    while size % Px != 0:
        Px -= 1
    Py = size // Px
    
    if Px * Py != size:
        raise ValueError("Could not factor worker size into a Px x Py grid")
    if L % Px != 0 or L % Py != 0:
        raise ValueError(f"Grid size L={L} must be divisible by Px={Px}, Py={Py}")
    
    lx, ly = L // Px, L // Py
    return Px, Py, lx, ly

def _load_initial_grid(path: str) -> np.ndarray:
    """Loads and validates the grid (Run only on Rank 0)."""
    grid = np.loadtxt(path, delimiter=",", dtype=np.int16)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError(f"input grid must be square 2D, got shape {grid.shape}")

    values = set(np.unique(grid).tolist())
    if values.issubset({0, 1}):
        grid = (2 * grid - 1).astype(np.int8, copy=False)
    elif values.issubset({-1, 1}):
        grid = grid.astype(np.int8, copy=False)
    else:
        raise ValueError(f"input grid must contain only -1/+1 or 0/1, got {sorted(values)}")

    return np.ascontiguousarray(grid)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_time = time.time()
    
    # --- Step 1: Configuration & Data Distribution (Rank 0 -> All) ---
    
    config = None 
    global_grid = None 

    if rank == 0:
        # Load data only on Master
        global_grid = _load_initial_grid("input.csv")
        L = int(global_grid.shape[0])
        Px, Py, lx, ly = _compute_process_grid(L, size)
        config = (L, Px, Py, lx, ly)
    
    # Broadcast configuration to all workers
    L, Px, Py, lx, ly = comm.bcast(config, root=0)
    
    # Prepare local block buffer
    initial_block = np.empty((lx, ly), dtype=np.int8)
    
    if rank == 0:
        # Send blocks to other ranks
        for r in range(size):
            rpx, rpy = r % Px, r // Px
            x0, x1 = rpx * lx, (rpx + 1) * lx
            y0, y1 = rpy * ly, (rpy + 1) * ly
            
            block_to_send = np.ascontiguousarray(global_grid[x0:x1, y0:y1])
            
            if r == 0:
                initial_block = block_to_send
            else:
                comm.Send(block_to_send, dest=r, tag=77)
    else:
        # Workers receive their block
        comm.Recv(initial_block, source=0, tag=77)

    # --- Step 2: Simulation Setup ---
    
    beta = 1.0 / (k_B * T)
    per_rank_rng = np.random.default_rng(seed + 1000003 * rank)
    common_rng = np.random.default_rng(seed)

    # Find this process's coordinates
    px, py = rank % Px, rank // Px

    # Local spins with ghost borders
    spins = np.empty((lx + 2, ly + 2), dtype=np.int8)
    spins[1:lx+1, 1:ly+1] = initial_block

    # Neighbor ranks
    up = (py - 1) % Py * Px + px
    down = (py + 1) % Py * Px + px
    left = py * Px + (px - 1) % Px
    right = py * Px + (px + 1) % Px

    # Preallocate buffers
    send_left = np.empty(lx, dtype=spins.dtype)
    send_right = np.empty(lx, dtype=spins.dtype)
    recv_left = np.empty(lx, dtype=spins.dtype)
    recv_right = np.empty(lx, dtype=spins.dtype)

    def exchange_ghosts():
        TAG_ROW_UP = 10
        TAG_ROW_DOWN = 11
        TAG_COL_LEFT = 20
        TAG_COL_RIGHT = 21

        # Rows
        comm.Sendrecv(sendbuf=spins[1, 1:ly+1], dest=up, sendtag=TAG_ROW_UP,
                      recvbuf=spins[lx+1, 1:ly+1], source=down, recvtag=TAG_ROW_UP)
        comm.Sendrecv(sendbuf=spins[lx, 1:ly+1], dest=down, sendtag=TAG_ROW_DOWN,
                      recvbuf=spins[0, 1:ly+1], source=up, recvtag=TAG_ROW_DOWN)

        # Columns
        send_left[:] = spins[1:lx+1, 1]
        send_right[:] = spins[1:lx+1, ly]

        comm.Sendrecv(sendbuf=send_left, dest=left, sendtag=TAG_COL_LEFT,
                      recvbuf=recv_right, source=right, recvtag=TAG_COL_LEFT)
        comm.Sendrecv(sendbuf=send_right, dest=right, sendtag=TAG_COL_RIGHT,
                      recvbuf=recv_left, source=left, recvtag=TAG_COL_RIGHT)

        spins[1:lx+1, 0] = recv_left
        spins[1:lx+1, ly+1] = recv_right

    # Metropolis lookup table
    exp_lookup = {
        4 * J: float(np.exp(-beta * (4 * J))),
        8 * J: float(np.exp(-beta * (8 * J))),
    }

    # Staleness tracking
    stale_top = np.zeros(ly, dtype=bool)
    stale_bottom = np.zeros(ly, dtype=bool)
    stale_left = np.zeros(lx, dtype=bool)
    stale_right = np.zeros(lx, dtype=bool)

    exchange_ghosts()

    # --- Step 3: Simulation Loop ---
    out = None
    if rank == 0:
        out = open('classic_magnetization.txt', 'w', buffering=1024 * 1024)
        
    for sweep in range(sweeps):
        for _ in range(lx * ly):
            i = int(common_rng.integers(1, lx + 1))
            j = int(common_rng.integers(1, ly + 1))

            is_border = (i == 1) or (i == lx) or (j == 1) or (j == ly)

            if is_border:
            
                local_need_sync = (
                    (i == 1 and stale_top[j - 1]) or
                    (i == lx and stale_bottom[j - 1]) or
                    (j == 1 and stale_left[i - 1]) or
                    (j == ly and stale_right[i - 1])
                )

                global_need_sync = comm.allreduce(local_need_sync, op=MPI.LOR)
                
                if global_need_sync:
                    exchange_ghosts()
                    stale_top[:] = False
                    stale_bottom[:] = False
                    stale_left[:] = False
                    stale_right[:] = False

            s = spins[i, j]
            nn = spins[i+1, j] + spins[i-1, j] + spins[i, j+1] + spins[i, j-1]
            dE = 2 * J * s * nn

            if dE <= 0:
                spins[i, j] = -s
            else:
                p = exp_lookup.get(int(dE), float(np.exp(-beta * dE)))
                if per_rank_rng.random() < p:
                    spins[i, j] = -s

            if is_border:
                if j == ly: stale_left[i - 1] = True
                if j == 1: stale_right[i - 1] = True
                if i == lx: stale_top[j - 1] = True
                if i == 1: stale_bottom[j - 1] = True

        # Calculate Magnetization
        local_M = np.sum(spins[1:lx+1, 1:ly+1])
        M = comm.reduce(local_M, op=MPI.SUM, root=0)
        
        if rank == 0:
            mag_per_spin = M / (L * L)
            out.write(f"{sweep} {mag_per_spin:.6f}\n")

    if rank == 0:
        out.close()

    # --- Step 4: Gather Results ---
    
    final_local = np.ascontiguousarray(spins[1:lx+1, 1:ly+1])
    gathered = None
    if rank == 0:
        gathered = np.empty((size, lx, ly), dtype=np.int8)
    
    # Gather all blocks to rank 0
    comm.Gather(final_local, gathered, root=0)

    if rank == 0:
        final_global = np.empty((L, L), dtype=np.int8)
        for r in range(size):
            rpx, rpy = r % Px, r // Px
            x0, x1 = rpx * lx, (rpx + 1) * lx
            y0, y1 = rpy * ly, (rpy + 1) * ly
            final_global[x0:x1, y0:y1] = gathered[r, :, :]

        end_time = time.time()
        elapsed = end_time - start_time
            
        np.savetxt("output.csv", final_global, fmt="%d", delimiter=",")
        print(f"Simulation completed in {elapsed:.6f} seconds")
        with open("results.csv", "a", encoding="utf-8") as f:
            f.write(f"{size},{elapsed:.6f}\n")

if __name__ == "__main__":
    main()