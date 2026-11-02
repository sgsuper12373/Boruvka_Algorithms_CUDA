# 23-10-2024

## For OpenMP code
- Create graphs for Uminho and our code for comparision
- 1,2,4,8,16, 24 all hyperthreading disaables and then 32 and 48 with hyperthreading enabled only in same NUMA node 
- Use both the NUMA nodes and do experiments without hyperthreading(16,24,32,48) and with hyperthreading enabled(1,2,4,8,16,24,32,48.(2 more config).96)
- Find the number of cores and other hardware config (memory,cache,shared mem size)

## For CUDA 
- Confirm the cuda code is correct for all the inputs, 
- compare the throughput compared the the openMP best throughput
- OPTIMZATIONS:  
    - use warp level and block level and hybrid mode 
    - avoid atomics to global memory by all threads ( only representative of group update the global memory (warp,block))
    - figure out if it's useful to exploit shared memory
    - compute time for each kernels and see which is taking most part of the overall time. // used nsysn 



# What results we want for comparision with UMINHO and My Boruvika
- 1 2 4 8 16, 24 on single NUMA node with hyper threading disabled
- 1 2 4 8 16 24 32 48 on single NUMA node with hyperthreading enabled
- Both NUMA node with 1 2 4 8 16 24 32 48 threads 
- Both NUMA nodes with with hyperthreading and 1 2 4 8 16 24 32 48 64 80 96


# 28-10-2026
- Warp and block optimzation
- 9 runs with median and other comparision 
- time the same thing as ecl-mst
- avoid atomics to global memory by all threads ( only representative of group update the global memory (warp,block))
- galois if possible 




- rewrote cheapest_find_kernel in `Boruvka_intermediate_warp_v1.cu` as **one warp per vertex**:
    - warp number = vertex u, lane = threadIdx.x % 32
    - lane k looks at edges k, k+32, k+64 ... of u, so a high degree vertex is shared by 32 threads instead of one thread doing everything (less thread divergence / load imbalance)
    - every lane keeps its own cheapest edge, then a normal warp min reduction with `__shfl_down_sync` (16, 8, 4, 2, 1). 
    - lane 0 marks u inactive if nothing is outgoing, otherwise does the check-then-atomicMin on cheapest[rep_u]
    - the find kernel now launches nodes*32 threads (`findBlocks`), the other kernels still use one thread per vertex
    - TPB must be a multiple of 32 now (a warp should never be split between two blocks)


## The three versions I am comparing now
| version | source | executable | script | job name |
|---|---|---|---|---|
| baseline | src/Boruvka_intermediate_cuda.cu | bin/mst | run_gpu_mst.sbatch | boruvka_base |
| atomic guard (read cheapest first, atomicMin only if cheaper) | src/Boruvka_intermediate_atomic_guard.cu | bin/mst_atomic_guard | run_atomic_guard.sbatch | boruvka_atomic_guard |
| warp per vertex | src/Boruvka_intermediate_warp_v1.cu | bin/mst_warp | run_warp_mst.sbatch | boruvka_warp |
| ECL-MST (reference) | ../ECL-MST/source/ECL-MST_10.cu | bin/ecl_mst | run_ecl_mst.sbatch | ecl_mst |


- Added a Makefile in Boruvka_Algorithms_CUDA/ that builds all four executables (`-O3 -arch=sm_90`).

## 9 runs + median 
- All the scripts now run every graph 9 times (`RUNS=9`) and write every run in a CSV:
    - our code: `reports/<job id>/boruvka_runs.csv` (graph, nodes, edges, run, binary, loop_seconds, throughput, verify)
    - ECL: `reports/ecl_<job id>/ecl_throughput.csv`
- `verify` is ok only if both the MST weight and the MST edge count match the CPU serial_half.
- The `binary` column has the executable name + its build time, so I know which build made which report.
- nsys still runs once per graph (after the 9 timed runs) for the per kernel breakdown.
- Increased the time limits: 6 h for our scripts, 4 h for ECL (the CPU check runs on every run).
- We time the same thing as ecl-mst: our "Boruvka loop" and ECL's "Dev" time both exclude the host<->device copies.



## Why warp per vertex is slower on some graphs
- The GPU issues a fixed number of warp instructions per cycle (4 schedulers per SM, 1 warp instruction each). An instruction costs the same slot whether 32 lanes are busy or only 2. So what matters is the TOTAL number of warp instructions, not how fast one warp finishes.
- europe_osm (degree ~2): base = 1.6M warps doing ~3 loop steps each. Warp version = 50.9M warps doing 1 step each, with 30 of 32 lanes idle. Every warp is faster, but there are 32x more of them, so ~10x more total work 
- Kernel time ≈ max( total work / GPU throughput , time of the slowest single warp ).
    - low degree graphs: no slow warp, so total work decides -> warp version loses
    - hub graphs (as-skitter, in-2004, soc-LiveJournal1): in base, ONE thread walks the hub's whole edge list (20k-35k edges, one memory read after another) and the round waits for it. Base find time per round / max degree ≈ 120-165 ns per edge, i.e. the round is basically that one thread. Warp version splits the hub over 32 lanes -> 4-10x faster.
- Average degree alone doesn't predict it (r4-2e23 deg 8 loses, citationCiteseer deg 8.6 wins). What matters is how uneven the degrees are inside a group of 32 vertices vs whether vertices have enough edges to fill 32 lanes (warp wastes lanes).

