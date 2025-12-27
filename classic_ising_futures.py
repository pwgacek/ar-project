#!/usr/bin/env python


import time
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, get_comm_workers
import numpy as np

T = 2.5
J = 1.0
sweeps = 20
seed = 123
k_B = 1.0


def _compute_process_grid(L: int, size: int) -> tuple[int, int, int, int]:
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



def metropolis_ising_parallel(args):
    comm = get_comm_workers()
    rank = comm.Get_rank()
    size = comm.Get_size()

    L, Px, Py, lx, ly, initial_block = args

    
    beta = 1.0 / (k_B * T)
    per_rank_rng = np.random.default_rng(seed + 1000003 * rank)
    common_rng = np.random.default_rng(seed)

    # --- Find this process's block ---
    px, py = rank % Px, rank // Px

    if initial_block.shape != (lx, ly):
        raise ValueError(f"initial_block must have shape ({lx}, {ly}), got {initial_block.shape}")

    # --- Local spins, with ghost borders ---
    spins = np.empty((lx + 2, ly + 2), dtype=np.int8)
    spins[1:lx+1, 1:ly+1] = initial_block

    # Neighbor ranks in the Px x Py process grid (periodic)
    up = (py - 1) % Py * Px + px
    down = (py + 1) % Py * Px + px
    left = py * Px + (px - 1) % Px
    right = py * Px + (px + 1) % Px

    # Preallocate contiguous buffers for halo exchange (avoid per-call allocations)
    send_left = np.empty(lx, dtype=spins.dtype)
    send_right = np.empty(lx, dtype=spins.dtype)
    recv_left = np.empty(lx, dtype=spins.dtype)
    recv_right = np.empty(lx, dtype=spins.dtype)

    def exchange_ghosts():
        # Paired exchanges prevent deadlock when neighbors are different ranks.
        # Also use tags to avoid message mixups when up==down or left==right.
        TAG_ROW_UP = 10
        TAG_ROW_DOWN = 11
        TAG_COL_LEFT = 20
        TAG_COL_RIGHT = 21

        # Rows (contiguous):
        # send our top interior row to UP, receive our bottom ghost row from DOWN
        comm.Sendrecv(
            sendbuf=spins[1, 1:ly+1], dest=up, sendtag=TAG_ROW_UP,
            recvbuf=spins[lx+1, 1:ly+1], source=down, recvtag=TAG_ROW_UP,
        )
        # send our bottom interior row to DOWN, receive our top ghost row from UP
        comm.Sendrecv(
            sendbuf=spins[lx, 1:ly+1], dest=down, sendtag=TAG_ROW_DOWN,
            recvbuf=spins[0, 1:ly+1], source=up, recvtag=TAG_ROW_DOWN,
        )

        # Columns (must be contiguous buffers for mpi4py)
        # Fill preallocated send buffers
        send_left[:] = spins[1:lx+1, 1]
        send_right[:] = spins[1:lx+1, ly]

        # # send our left interior col to LEFT, receive our right ghost col from RIGHT
        comm.Sendrecv(
            sendbuf=send_left, dest=left, sendtag=TAG_COL_LEFT,
            recvbuf=recv_right, source=right, recvtag=TAG_COL_LEFT,
        )
        # send our right interior col to RIGHT, receive our left ghost col from LEFT
        comm.Sendrecv(
            sendbuf=send_right, dest=right, sendtag=TAG_COL_RIGHT,
            recvbuf=recv_left, source=left, recvtag=TAG_COL_RIGHT,
        )

        spins[1:lx+1, 0] = recv_left
        spins[1:lx+1, ly+1] = recv_right

    # Precompute Metropolis acceptance for possible dE values in 2D Ising (nearest neighbors)
    # nn in {-4,-2,0,2,4}, s in {-1,1} => dE in {-8J,-4J,0,4J,8J}
    # We only need probabilities for positive dE.
    exp_lookup = {
        4 * J: float(np.exp(-beta * (4 * J))),
        8 * J: float(np.exp(-beta * (8 * J))),
    }
    
    if rank == 0:
        out = open('classic_magnetization.txt', 'w', buffering=1024 * 1024)

    # Ghost staleness tracking (potentially out-of-date since last sync)
    stale_top = np.zeros(ly, dtype=bool)
    stale_bottom = np.zeros(ly, dtype=bool)
    stale_left = np.zeros(lx, dtype=bool)
    stale_right = np.zeros(lx, dtype=bool)

    # Ensure ghost borders are initially correct
    exchange_ghosts()

    is_border_counter = 0
    is_not_border_counter = 0
    is_communication_counter = 0

    for sweep in range(sweeps):
        for _ in range(lx * ly):
            i = int(common_rng.integers(1, lx + 1))
            j = int(common_rng.integers(1, ly + 1))

            is_border = (i == 1) or (i == lx) or (j == 1) or (j == ly)

            if is_border:
                if rank == 0:
                    is_border_counter += 1
                # If the proposed update needs potentially stale ghost data, we must sync.
                local_need_sync = (
                    (i == 1 and stale_top[j - 1]) or
                    (i == lx and stale_bottom[j - 1]) or
                    (j == 1 and stale_left[i - 1]) or
                    (j == ly and stale_right[i - 1])
                )

                # Synchronization between all processors takes place
                global_need_sync = comm.allreduce(local_need_sync, op=MPI.LOR)
                if global_need_sync:
                    is_communication_counter += 1
                    exchange_ghosts()
                    stale_top[:] = False
                    stale_bottom[:] = False
                    stale_left[:] = False
                    stale_right[:] = False

            else:
                if rank == 0:
                    is_not_border_counter += 1
            s = spins[i, j]
            nn = (
                spins[i + 1, j] + spins[i - 1, j] +
                spins[i, j + 1] + spins[i, j - 1]
            )
            dE = 2 * J * s * nn
            if dE <= 0:
                spins[i, j] = -s
            else:
                p = exp_lookup.get(int(dE), float(np.exp(-beta * dE)))
                if per_rank_rng.random() < p:
                    spins[i, j] = -s

            # Update "potentially stale" markers based only on synchronized site selection.
            # This is conservative: a selected neighbor-border site might have flipped.
            if is_border:
                if j == ly:
                    stale_left[i - 1] = True      # left neighbor's right border
                if j == 1:
                    stale_right[i - 1] = True     # right neighbor's left border
                if i == lx:
                    stale_top[j - 1] = True       # up neighbor's bottom border
                if i == 1:
                    stale_bottom[j - 1] = True    # down neighbor's top border

        # --- Gather magnetization after each sweep ---
        local_M = np.sum(spins[1:lx+1, 1:ly+1])
        M = comm.reduce(local_M, op=MPI.SUM, root=0)
        if rank == 0:
            mag_per_spin = M / (L * L)
            out.write(f"{sweep} {mag_per_spin:.6f}\n")

    if rank == 0:
        print(f"percent border accesses: {is_border_counter / (is_border_counter + is_not_border_counter) * 100:.2f}%")
        print(f"percent communication steps: {is_communication_counter / (is_border_counter + is_not_border_counter) * 100:.2f}%")
        out.close()

    # Gather final spins (without ghost borders) onto worker-rank 0 and assemble full grid.
    final_local = np.ascontiguousarray(spins[1:lx+1, 1:ly+1])
    if rank == 0:
        gathered = np.empty((size, lx, ly), dtype=np.int8)
    else:
        gathered = None

    comm.Gather(final_local, gathered, root=0)

    if rank == 0:
        final_global = np.empty((L, L), dtype=np.int8)
        for r in range(size):
            rpx, rpy = r % Px, r // Px
            x0, x1 = rpx * lx, (rpx + 1) * lx
            y0, y1 = rpy * ly, (rpy + 1) * ly
            final_global[x0:x1, y0:y1] = gathered[r, :, :]
        return final_global

    return None


if __name__ == "__main__":
    global_grid = _load_initial_grid("input.csv")
    start_time = time.time()

    L = int(global_grid.shape[0])

    with MPIPoolExecutor() as executor:
        nworkers = executor.num_workers
        Px, Py, lx, ly = _compute_process_grid(L, nworkers)

        task_args = []
        for r in range(nworkers):
            rpx, rpy = r % Px, r // Px
            x0, x1 = rpx * lx, (rpx + 1) * lx
            y0, y1 = rpy * ly, (rpy + 1) * ly
            block = np.ascontiguousarray(global_grid[x0:x1, y0:y1].astype(np.int8, copy=False))
            task_args.append((L, Px, Py, lx, ly, block))

        
        results = list(executor.map(metropolis_ising_parallel, task_args))
        end_time = time.time()

    elapsed = end_time - start_time
    output_grid = next((r for r in results if r is not None), None)
    np.savetxt("output.csv", output_grid, fmt="%d", delimiter=",")
    print(f"Simulation completed in {elapsed:.6f} seconds")
    with open("results.csv", "a", encoding="utf-8") as f:
        f.write(f"{nworkers},{elapsed:.6f}" + "\n")
