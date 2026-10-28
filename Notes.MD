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
    - compute time for each kernels and see which is taking most part of the overall time.



# What results we want for comparision with UMINHO and My Boruvika
- 1 2 4 8 16, 24 on single NUMA node with hyper threading disabled, 
- 1 2 4 8 16 24 32 48 on single NUMA node with hyperthreading enabled
- Both NUMA node with 1 2 4 8 16 24 32 48 threads 
- Both NUMA nodes with with hyperthreading and 1 2 4 8 16 24 32 48 64 80 96