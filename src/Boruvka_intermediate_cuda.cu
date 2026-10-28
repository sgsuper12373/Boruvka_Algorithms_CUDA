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



/**
 * @brief This methods will find the parent of node v
 * 		  not read only implementation, path contration will be done here 
 * 		  just adding for weight verification. 
 * @todo implement host_find for cpu verification. 
 * @param parents 
 * @param v 
 * @return __host__ 
 */
__host__ int host_find( const vector<int>& parents, const int v ){
	return 0; 
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
 * @brief 
 * Find the cheapest outgoing edge from each component and store it's edge index in cheapest[comp]; 
 *	assign the thread per vertex u, each thread will find it's parent and then travese the corresponding edge list for d_nlist 
 *	in each neighbour v in d_nlist find it's parent, 
 *	if parent(u) == parent(v) then they are in same component, skip this entry 
 *	else check if there already exist some cheapest edge (might get updated by the other threads
 *	if current edge is cheaper than existing edge in d_cheapest the update it usign atomicMin
 * 
 * @param parents 
 * @param cheapest 
 * @param G 
 * @return __global__ 
 */
__global__ void cheapest_find_kernel( int* parents, ull* cheapest,bool* isActive, ECLgraph G ){


	// u is basicall tid
	int u = threadIdx.x + blockDim.x * blockIdx.x; 
	
	// size of parents array will always equal to G.nodes
	// continue only if the vertex is active. 
	if( u < G.nodes && isActive[u]){
		int rep_u, v, rep_v;  					// store the representatives of u and v
		rep_u = device_find(parents, u); 	// find the representative of u
		ull best, key; 
		best = INF; 					// initial best outgoing edge, compute locally then update globally 
		bool has_outgoing = false;  		// If this vertex have no outgoing edge mark it inactive so next time we we can skip it. 


		// traverse the edge list, and update the cheapest edge array 
		for( int i = G.nindex[u] ; i < G.nindex[u+1]; i++ ){
			v = G.nlist[i]; 
			rep_v = device_find(parents, v); 
			if( rep_u == rep_v ) continue; 
			has_outgoing = true; 

			key = edgeKey(G.eweight[i], i ); 
			best = best < key ? best : key;  // update the local cheapest edge. 

		}

		// update if it have outgoing edge, else mark inactive 
		// @note OPTIMIZATION: added atomic guard here 
		// if( has_outgoing ) {
		// 	if( best < cheapest[rep_u]) atomicMin(&cheapest[rep_u], best);
		// }
		if( has_outgoing) atomicMin(&cheapest[rep_u], best);
		else isActive[u] = false; 
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
double GPU_MST(ECLgraph G, bool*h_MST, ull* h_MST_Weight){

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
	int TPB = 256; 
	int Blocks = (G.nodes + TPB - 1 ) / TPB; 
	cudaMemset(d_isActive,1,G.nodes*sizeof(bool));
	init_parents_kernel<<<Blocks, TPB>>>(d_parent, G.nodes);
	CHECK();

	// memset + init kernel are async, finish them so they are not counted in the loop time
	cudaDeviceSynchronize();
	auto boruvka_start = chrono::steady_clock::now();
	while( h_changed ){

		// set h_changed and d_changed to false;
		h_changed = false;
		cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

		// reset the cheapest array;
		cudaMemset(d_cheapest,0xFF,G.nodes*sizeof(ull));

		// function for updating the cheapest array.
		// same config as init Parents works here (vertex centric) we have thread per vertex
		cheapest_find_kernel<<<Blocks, TPB>>>(d_parent,d_cheapest, d_isActive,d_G);
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input.egr>\n", argv[0]);
        return -1;
    }

    ECLgraph G = readECLgraph(argv[1]);
    printf("Graph: %s\nNodes: %d\nEdges: %d\n", argv[1], G.nodes, G.edges);

    if (G.eweight == NULL) {
        assign_random_weights(G, MAX_WEIGHT, WEIGHT_SEED);
        printf("Note: no weights in file, generated with seed %u\n", WEIGHT_SEED);
    }

    bool* h_MST = new bool[G.edges];
    ull   h_MST_Weight = 0;

    double boruvka_loop_seconds = GPU_MST(G, h_MST, &h_MST_Weight);

    // Host-side cross-check: recompute weight and edge count from h_MST
    long long check_weight = 0;
    int mst_edges = 0;
    for (int i = 0; i < G.edges; i++) {
        if (h_MST[i]) { check_weight += G.eweight[i]; mst_edges++; }
    }

    printf("MST weight (GPU)  : %lld\n", (long long)h_MST_Weight);
    printf("MST weight (check): %lld %s\n", check_weight,
           check_weight == (long long)h_MST_Weight ? "(match)" : "(MISMATCH)");
    printf("MST edges         : %d\n", mst_edges);
    printf("Boruvka loop      : %.6f s\n", boruvka_loop_seconds);
    printf("Throughput        : %.2f Medges/s\n",boruvka_loop_seconds > 0.0? G.edges / boruvka_loop_seconds / 1e6: 0.0);

    delete[] h_MST;
    freeECLgraph(G);
    return 0;
}

