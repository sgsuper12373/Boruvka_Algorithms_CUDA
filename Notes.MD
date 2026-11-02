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
    - boruvka intermediate code: `reports/<job id>/boruvka_runs.csv` (graph, nodes, edges, run, binary, loop_seconds, throughput, verify)
    - ECL: `reports/ecl_<job id>/ecl_throughput.csv`
- `verify` is ok only if both the MST weight and the MST edge count match the CPU serial_half.
- nsys still runs once per graph (after the 9 timed runs) for the per kernel breakdown.


## Why warp per vertex is slower on some graphs
- The SM have some fixed number of warp schedulers, The number of warps launched will matter the time taken by overall execution because if warp have low work like 2 neighbours evaluation then all 30 lines of warps sits idea
- at a single time some fixed number of warps can execute on the SM, increasing number of warps will result into more warps per SM and SM will execute them one after another. 
- europe_osm (degree ~2): base = 1.6M warps doing ~3 loop steps each. Warp version = 1.6 * 32 => 51.2M warps doing 1 step each, with 30 of 32 lanes idle. Every warp is faster, but there are 32x more of them, so  more total work 

- Average degree alone doesn't predict it (r4-2e23 deg 8 loses, citationCiteseer deg 8.6 wins). What matters is how uneven the degrees are inside a group of 32 vertices vs whether vertices have enough edges to fill 32 lanes 
