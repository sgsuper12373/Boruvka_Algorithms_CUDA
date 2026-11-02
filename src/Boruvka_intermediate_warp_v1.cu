#include<iostream>
#include<bits/stdc++.h>
#include<cuda.h> 
#include<cuda_runtime.h>
#include<ECLgraph.h>
#include <random>
#include <chrono>


using namespace std; 
typedef unsigned long long ull; 
constexpr ull INF = ~0ULL; 


static const int      MAX_WEIGHT  = 50;
static const uint32_t WEIGHT_SEED = 20260907u;


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
	fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
	exit(-1);
  }
}

#ifdef DEBUG
  #define CHECK() CheckCuda(__LINE__)
#else
  #define CHECK()
#endif



// ====================================================== HOST (CPU reference) Functions ======================================================

/**
 * @brief This methods will find the representative (root) of node v
 * 		  does the half path compression: every node on the path points to its grandparent
 * 		  used only for the CPU verification (serial_half).
 * @param parents
 * @param v
 * @return __host__
 */
__host__ int host_find( vector<int>& parents, int v ){
	while( parents[v] != v ){
		parents[v] = parents[parents[v]];
		v = parents[v];
	}
	return v;
}

/**
 * @brief union by size, smaller component is attached under the bigger one
 * 		  return false if u and v are already in the same component
 * @param parents
 * @param size
 * @param u
 * @param v
 * @return __host__
 */
__host__ bool host_union( vector<int>& parents, vector<int>& size, int u, int v ){
	int ult_u = host_find(parents, u);
	int ult_v = host_find(parents, v);

	// if in same comp return false;
	if( ult_u == ult_v ) return false;

	if( size[ult_u] > size[ult_v] ) swap(ult_u, ult_v);

	parents[ult_u] = ult_v;
	size[ult_v] += size[ult_u];
	return true;
}


// ====================================================== COMMON Funtions ======================================================

/**
 * @brief encodes the edge_weight and csr edge index and return ull 
 * 		 | Edge weight (32 bits) | CSR Edge index (32 bits) | 
 * @param weight 
 * @param edge_index 
 * @return __host__ 
 */
__host__ __device__ __forceinline__ ull edgeKey(int weight, int edge_index) {
    const unsigned int ordered_weight = static_cast<unsigned int>(weight) ^ 0x80000000u;
    return (static_cast<unsigned long long>(ordered_weight) << 32) |
           static_cast<unsigned int>(edge_index);
}




// ====================================================== DEVICE Functions   ======================================================





/**
 * @brief Returns the representative (root) of v.
 *        O(1) direct lookup: valid because compress_tree_kernel runs at the
 *        end of every round, so each vertex points directly at its root.	
 *        Read-only: called only from cheapest_find_kernel, during which no
 *        unions happen, so parents[] is stable.
 */
__device__ __forceinline__ int device_find(const int* parents, const int v) {
    return parents[v];
}

/**
 * @brief finds the parents of u and v,
 * 		  if( rep_u == rep_v) return false
 * 		  else continously try to merge 
 * 
 * @param parents 
 * @param u 
 * @param v 
 * @return __device__ 
 */
__device__ __forceinline__ bool device_union(int* parents, int u, int v) {
    volatile int* p = parents;
    while (true) {
        u = p[u]; v = p[v];
        if (u == v) return false;
        if (u > v) { int t = u; u = v; v = t; }
        if (atomicCAS(&parents[v], v, u) == v) return true;
    }
}


//============================================ KERNELS =============================================

/**
 * @brief Initialize the parents array allocated on the GPU 
 * 		  initially every node is parent of itself. 
 * 
 * @param parents 
 * @param N 
 * @return __global__ 
 */
__global__ void init_parents_kernel( int *parents, int N ){
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	if(gid >= N ) return ; 
	parents[gid] = gid; 
}

/**
 * @brief Find the cheapest outgoing edge from each component, ONE WARP PER VERTEX.
 *
 * 	Why: with one thread per vertex, a thread with a degree-2000 vertex loops 2000 times
 * 	while its warp neighbour with a degree-2 vertex sits idle (thread divergence / load imbalance).
 * 	Here the 32 lanes of a warp share the edges of one vertex u:
 *
 * 		lane 0 : edges start+0, start+32, start+64, ...
 * 		lane 1 : edges start+1, start+33, start+65, ...
 * 		...
 * 		lane 31: edges start+31, start+63, ...
 *
 * 	1. warp number = u, lane = position of the thread inside the warp
 * 	2. every lane keeps the cheapest outgoing edge (encoded key) among the edges it looked at
 * 	3. warp min reduction with __shfl_down_sync, after it lane 0 holds the min of all 32 lanes
 * 	   (all 32 lanes work on the same vertex, hence the same component, so a plain reduction is correct)
 * 	4. lane 0 marks u inactive if no outgoing edge is left, else updates cheapest[rep_u] with atomicMin
 *
 * @note needs blockDim.x to be a multiple of 32, so a warp never spans two blocks / two vertices
 * 	     and the full mask 0xffffffff is valid in the shuffle.
 * @param parents 
 * @param cheapest 
 * @param isActive 
 * @param G 
 * @return __global__ 
 */
__global__ void cheapest_find_kernel( int* parents, ull* cheapest, bool* isActive, ECLgraph G ){

	// one warp per vertex: warp number = vertex, lane = position inside the warp
	// long long: nodes * 32 threads can get close to the int limit 
	long long gid = threadIdx.x + (long long)blockDim.x * blockIdx.x;
	int u    = (int)(gid / 32);
	int lane = threadIdx.x % 32;

	// all 32 lanes of a warp have the same u, so the whole warp leaves together
	if( u >= G.nodes || !isActive[u] ) return;

	int rep_u = device_find(parents, u);

	// each lane looks at every 32nd edge of u and keeps its local cheapest edge
	ull best = INF;
	for( int i = G.nindex[u] + lane; i < G.nindex[u+1]; i += 32 ){
		int rep_v = device_find(parents, G.nlist[i]);
		if( rep_u == rep_v ) continue;		// same component, not an outgoing edge

		ull key = edgeKey(G.eweight[i], i );
		best = best < key ? best : key;
	}

	// warp min reduction: offsets 16, 8, 4, 2, 1
	// in every step lane k takes the value of lane k+offset, after the last step lane 0 has the min of the warp
	for( int offset = 16; offset > 0; offset /= 2 ){
		ull other = __shfl_down_sync(0xffffffff, best, offset);
		best = best < other ? best : other;
	}

	// only lane 0 writes, so one global atomic per vertex
	if( lane == 0 ){
		if( best == INF ) isActive[u] = false;		// no outgoing edge left, skip u in next rounds
		else if( best < cheapest[rep_u] ) atomicMin(&cheapest[rep_u], best);	// check first, atomic only if it can win
	}
}




/**
 * @brief Construct the MST using by doing the union,
 * 		  We might get the same edge (u-v, v-u) hence union should be done only once 
 * 		  The device_union function should return the false if u-v found to be in same comp to avoid 
 * 		  doubly adding weight to the MST
 * @param cheapest 
 * @param parents 
 * @param G 
 * @param inMST 
 * @param MST_Weight 
 * @param changed 
 * @return __global__ 
 */
__global__ void construct_mst_kernel(ull *cheapest,int* parents,ECLgraph G, bool *inMST, ull* MST_Weight, bool* changed){
	int u = threadIdx.x + blockDim.x*blockIdx.x; 
	if( u < G.nodes && cheapest[u] != INF ){
		int i = (int)(cheapest[u] & 0xffffffffu);
		int w = static_cast<int>((cheapest[u] >> 32) ^ 0x80000000u);
		int v = G.nlist[i];

		if(device_union(parents,u,v)){
			atomicAdd(MST_Weight,(ull)w); 
			inMST[i] = 1; 
			*changed = true; 
		}

	}
}




/**
 * @brief flattens the tree
 * 		 In this code we are not retraceing the path to update every node visited in the path because we have 
 * 		 thread per vertex, so there would be need to atomics if we want threads to read compressed path, 
 * @note 
 * Why compress_tree_kernel needs no atomics:
 *  1. No concurrent unions: it runs after construct_mst_kernel finishes
 *     (kernel boundary = global barrier), so roots are fixed during the kernel.
 *  2. Single writer per slot: thread u writes only parents[u]; no write-write races.
 *  3. Read-write races are benign: a reader of parents[x] sees either the old
 *     parent or x's root (aligned 32-bit accesses never tear). Both are
 *     ancestors on the same path, so the walk reaches the same root.
 *  4. Guaranteed termination: every parent pointer points to a strictly
 *     smaller ID (union hooks larger under smaller; compress writes the root,
 *     the tree's minimum ID), so each step decreases the ID and must stop at
 *     a root (parents[r] == r). This also rules out cycles.
 * @param parents 
 * @param N 
 * @return __global__ 
 */
__global__ void compress_tree_kernel(int* parents, int N) {
    int u = threadIdx.x + blockDim.x * blockIdx.x;
    if (u >= N) return;

    int root = parents[u];
    while (root != parents[root]) root = parents[root];   // walk up to the root

    if (parents[u] != root) parents[u] = root;            // skip redundant writes
}



// ========================================================= MST Computation functions ==============================================================================

/**
 * @brief
 *
 *
 * @param G ELCgraph containing data on host
 * @param inMST boolean vector to know if edge is in MST
 * @return time of the Boruvka loop in seconds (used for throughput),
 *         per-kernel / memcpy timing is taken from nsys (see scripts/run_gpu_mst.sbatch)
 */
double GPU_MST(ECLgraph G, bool*h_MST, ull* h_MST_Weight, int TPB){

	bool* d_MST = NULL;
	cudaMalloc(&d_MST, G.edges*sizeof(bool)); 
	cudaMemset(d_MST,0,G.edges*sizeof(bool));

	// parent array for which works as DSU here, device only for now 
	int* d_parent = NULL; 
	cudaMalloc(&d_parent, G.nodes*sizeof(int)); 

	// d_cheapsest array to hold cheapest outgoing edge from the node i, encoded edge hence datatype will be unsigned long long 
	ull* d_cheapest = NULL, *d_MST_Weight = NULL; 
	cudaMalloc(&d_cheapest, G.nodes*sizeof(ull)); 
	cudaMalloc(&d_MST_Weight,sizeof(ull)); 
	cudaMemset(d_MST_Weight,0, sizeof(ull)); 
	

	// Allocate memory for the ECL_Graph array on GPU
	int* d_nindex = NULL, *d_nlist = NULL, *d_eweight = NULL ; 
	bool* d_isActive = NULL; 
	cudaMalloc(&d_nindex, (G.nodes+1)*sizeof(int) ); 
	cudaMalloc(&d_nlist, G.edges*sizeof(int)); 
	cudaMalloc(&d_eweight, G.edges*sizeof(int)); 
	cudaMalloc(&d_isActive, G.nodes*sizeof(bool));
	bool* d_changed = NULL;
	cudaMalloc(&d_changed, sizeof(bool));

	// copy data from ECL_graph G to device pointers
	cudaMemcpy(d_nindex,G.nindex,(G.nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nlist, G.nlist, G.edges*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_eweight, G.eweight, G.edges*sizeof(int), cudaMemcpyHostToDevice);

	// Creating compact graph csr graph structure using the ECLgraph struct 
	ECLgraph d_G; 
	d_G.nodes = G.nodes; 
	d_G.edges = G.edges; 
	d_G.nindex = d_nindex; 
	d_G.nlist = d_nlist; 
	d_G.eweight = d_eweight; 
	

	// now data is ready on GPU we can start the MST computation loop, 
	// varaible to decide when to terminate the algorithsm 
	bool h_changed = true; 
	cudaMemset(d_changed, 1, sizeof(bool)); // set as true before starting of roud, at loop begining set to false again. 


	// some basic kernels to initialize the d_parent, d_isactive and d_cheapest array
	// parent array, d_isActive initialization is just one time. but d_cheapest needs to reset after every round its done using the cudaMemset
	int Blocks = (G.nodes + TPB - 1 ) / TPB; 
	// cheapest_find_kernel uses one warp (32 threads) per vertex, so it needs 32x more threads
	int findBlocks = (int)(((long long)G.nodes * 32 + TPB - 1) / TPB);
	cudaMemset(d_isActive,1,G.nodes*sizeof(bool));
	// finish the memset first, it is setup and not part of the timed Boruvka loop
	cudaDeviceSynchronize();

	// Boruvka loop time starts here: init_parents_kernel + every round are counted,
	// the allocations, H2D copies and the isActive memset above are not
	auto boruvka_start = chrono::steady_clock::now();
	init_parents_kernel<<<Blocks, TPB>>>(d_parent, G.nodes);
	CHECK();

	// wait for init_parents_kernel before the first round
	cudaDeviceSynchronize();
	while( h_changed ){

		// set h_changed and d_changed to false;
		h_changed = false;
		cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

		// reset the cheapest array;
		cudaMemset(d_cheapest,0xFF,G.nodes*sizeof(ull));

		// function for updating the cheapest array.
		// warp per vertex: findBlocks launches 32 threads for every vertex
		cheapest_find_kernel<<<findBlocks, TPB>>>(d_parent,d_cheapest, d_isActive,d_G);
		CHECK();


		// write here kernle to construct the MST and unify the components using the cheapest outgoing edge
		// here we will traverse the cheapest array and decide if we want to include the edge in the MST or not
		// for launching the kernel we can use the same config as init_parents
		construct_mst_kernel<<<Blocks,TPB>>>(d_cheapest, d_parent, d_G, d_MST, d_MST_Weight, d_changed);
		CHECK();

		// compress tree kernel -> flattens the whole tree(parents array);
		// same config as init_parents (vertex centric)
		compress_tree_kernel<<<Blocks, TPB>>>(d_parent, G.nodes);
		CHECK();

		// get the value of d_changed
		cudaMemcpy(&h_changed, d_changed, sizeof(bool),cudaMemcpyDeviceToHost);
	}
	// the blocking memcpy above means all GPU work of the loop is done here
	double boruvka_loop_seconds = chrono::duration<double>(chrono::steady_clock::now() - boruvka_start).count();


	// copy results back to host
	cudaMemcpy( h_MST, d_MST, G.edges*sizeof(bool), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_MST_Weight, d_MST_Weight, sizeof(ull), cudaMemcpyDeviceToHost );

	CHECK(); 
	
	// free the device memory 
	cudaFree(d_MST); 
	cudaFree(d_changed); 
	cudaFree(d_cheapest); 
	cudaFree(d_parent); 
	cudaFree(d_nindex); 
	cudaFree(d_nlist); 
	cudaFree(d_eweight); 
	cudaFree(d_isActive); 
	cudaFree(d_MST_Weight);

	return boruvka_loop_seconds;
}


/**
 * @brief CPU reference MST, port of serial_half (Boruvka_CPU<DSU_half_cpu>) from
 * 		  Summer_Prjoect_2026/Boruvika_algo/src/ECL_boruvkas.cpp
 * 		  used to verify the MST weight computed on the GPU.
 * @param G ELCgraph containing data on host
 * @param mst_edges number of edges added to the MST (nodes - components)
 * @return MST weight
 */
long long CPU_MST_serial_half(ECLgraph G, int* mst_edges){
	vector<int> parents(G.nodes), size(G.nodes, 1);
	iota(parents.begin(), parents.end(), 0);

	long long MST_Weight = 0;
	int prev_comps = INT_MAX;
	int curr_comps = G.nodes;

	vector<int> comp(G.nodes);     // comp[u] = component id of node u
	vector<int> cheapest(G.nodes); // cheapest[c] = edge index of comp c's cheapest outgoing edge

	while( prev_comps != curr_comps ){
		prev_comps = curr_comps;

		// PHASE_0: flatten component ids once and reset cheapest
		for( int u = 0 ; u < G.nodes; u++ ){
			comp[u] = host_find(parents, u);
			cheapest[u] = INT_MAX;
		}

		// PHASE_1: find the cheapest outgoing edge per component
		for( int u = 0 ; u < G.nodes; u++ ){
			int ult_u = comp[u];
			for( int i = G.nindex[u]; i < G.nindex[u+1]; i++ ){
				int v = G.nlist[i];
				int w = G.eweight[i];

				if( ult_u == comp[v] ) continue; // same comp, skip

				if( cheapest[ult_u] == INT_MAX || w < G.eweight[ cheapest[ult_u] ] ){
					cheapest[ult_u] = i;
				}
			}
		}

		// PHASE_2: merge comps
		for( int u = 0 ; u < G.nodes; u++ ){
			if( cheapest[u] == INT_MAX ) continue;

			int i = cheapest[u];
			int v = G.nlist[i];
			int w = G.eweight[i];

			if( host_union(parents, size, u, v) ){
				MST_Weight += w;
				curr_comps--;
			}
		}
	}

	*mst_edges = G.nodes - curr_comps;
	return MST_Weight;
}


/**
 * @brief Same logic as ECL_boruvkas.cpp: one random weight per undirected edge
 *        (u <= v), mirrored to the reverse CSR entry v -> u.
 */
void assign_random_weights(ECLgraph& G, int max_weight, uint32_t seed) {
    if (G.edges == 0) { G.eweight = NULL; return; }

    G.eweight = static_cast<int*>(malloc(G.edges * sizeof(*G.eweight)));
    if (G.eweight == NULL) {
        fprintf(stderr, "ERROR: memory allocation failed while assigning weights\n");
        exit(EXIT_FAILURE);
    }
    fill(G.eweight, G.eweight + G.edges, 0);

    mt19937 generator(seed);
    uniform_int_distribution<int> distribution(1, max_weight);

    for (int u = 0; u < G.nodes; ++u) {
        for (int i = G.nindex[u]; i < G.nindex[u + 1]; ++i) {
            const int v = G.nlist[i];
            if (u > v || G.eweight[i] != 0) continue;

            const int weight = distribution(generator);
            G.eweight[i] = weight;
            if (u == v) continue;

            for (int j = G.nindex[v]; j < G.nindex[v + 1]; ++j) {
                if (G.nlist[j] == u && G.eweight[j] == 0) { G.eweight[j] = weight; break; }
            }
        }
    }
}


void print_usage(){
	fprintf(stderr, "Usage: ./executable <input.egr>\n" );
	fprintf(stderr, "Other argument options available: \n"); 
	fprintf(stderr, "\t --TPB <Threads per block>   (multiple of 32, 32 to 1024, default 256)\n");

}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

	int TPB = 256;   // default threads per block
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--TPB" && i + 1 < argc) {
            TPB = atoi(argv[++i]);
			// multiple of 32: cheapest_find_kernel gives one full warp to each vertex
			if( TPB <= 0 || TPB > 1024 || TPB % 32 != 0 ){
				fprintf(stderr, "ERROR: Threads per block should be a multiple of 32 between 32 and 1024\n");
				return 1;
			}
        } else {
            cerr << "Unknown or misplaced argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    ECLgraph G = readECLgraph(argv[1]);
    printf("Graph: %s\nNodes: %d\nEdges: %d\n", argv[1], G.nodes, G.edges);
    printf("Threads per block : %d\n", TPB);

    if (G.eweight == NULL) {
        assign_random_weights(G, MAX_WEIGHT, WEIGHT_SEED);
        printf("Note: no weights in file, generated with seed %u\n", WEIGHT_SEED);
    }

    bool* h_MST = new bool[G.edges];
    ull   h_MST_Weight = 0;

    double boruvka_loop_seconds = GPU_MST(G, h_MST, &h_MST_Weight, TPB);

    // number of MST edges marked by the GPU
    int mst_edges = 0;
    for (int i = 0; i < G.edges; i++) {
        if (h_MST[i]) mst_edges++;
    }

    // CPU reference (serial_half) for verification
    int cpu_mst_edges = 0;
    auto cpu_start = chrono::steady_clock::now();
    long long cpu_weight = CPU_MST_serial_half(G, &cpu_mst_edges);
    double cpu_seconds = chrono::duration<double>(chrono::steady_clock::now() - cpu_start).count();

    printf("MST weight (GPU)  : %lld\n", (long long)h_MST_Weight);
    printf("MST weight (CPU)  : %lld %s\n", cpu_weight, cpu_weight == (long long)h_MST_Weight ? "(match)" : "(MISMATCH)");
    printf("MST edges (GPU)   : %d\n", mst_edges);
    printf("MST edges (CPU)   : %d %s\n", cpu_mst_edges, cpu_mst_edges == mst_edges ? "(match)" : "(MISMATCH)");
    printf("CPU serial_half   : %.6f s\n", cpu_seconds);
    printf("Boruvka loop      : %.6f s\n", boruvka_loop_seconds);
    printf("Throughput        : %.2f Medges/s\n",boruvka_loop_seconds > 0.0? G.edges / boruvka_loop_seconds / 1e6: 0.0);

    delete[] h_MST;
    freeECLgraph(G);
    return 0;
}

