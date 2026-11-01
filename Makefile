# ---------------------------------------------------------------------------
# Builds every MST executable used by the sbatch scripts in scripts/.
#
# Run on the Madhava LOGIN node (compute nodes can't compile), from
# Boruvka_Algorithms_CUDA/:
#   module load cuda-12.4 gcc-12.4.0
#   make                    build all executables
#   make bin/mst_warp       build only one of them
#   make clean              delete all executables
#
#   executable              source                                     used by
#   bin/mst                 src/Boruvka_intermediate_cuda.cu           run_gpu_mst.sbatch
#   bin/mst_atomic_guard    src/Boruvka_intermediate_atomic_guard.cu   run_atomic_guard.sbatch
#   bin/mst_warp            src/Boruvka_intermediate_warp_v1.cu        run_warp_mst.sbatch
#   bin/ecl_mst             ../ECL-MST/source/ECL-MST_10.cu            run_ecl_mst.sbatch
#
# make only rebuilds an executable when its .cu (or ECLgraph.h) changed.
# Don't add -DDEBUG for timing runs: it syncs after every kernel.
# ---------------------------------------------------------------------------

NVCC  = nvcc
# sm_90 = H100
FLAGS = -O3 -arch=sm_90

ECL_DIR = ../ECL-MST/source

all: bin/mst bin/mst_atomic_guard bin/mst_warp bin/ecl_mst

# $< = first file after the colon (the .cu), $@ = the executable being built
bin/mst: src/Boruvka_intermediate_cuda.cu src/ECLgraph.h
	mkdir -p bin
	$(NVCC) $(FLAGS) -Isrc $< -o $@

bin/mst_atomic_guard: src/Boruvka_intermediate_atomic_guard.cu src/ECLgraph.h
	mkdir -p bin
	$(NVCC) $(FLAGS) -Isrc $< -o $@

bin/mst_warp: src/Boruvka_intermediate_warp_v1.cu src/ECLgraph.h
	mkdir -p bin
	$(NVCC) $(FLAGS) -Isrc $< -o $@

bin/ecl_mst: $(ECL_DIR)/ECL-MST_10.cu $(ECL_DIR)/ECLgraph.h
	mkdir -p bin
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f bin/mst bin/mst_atomic_guard bin/mst_warp bin/ecl_mst

# these are commands, not files
.PHONY: all clean
