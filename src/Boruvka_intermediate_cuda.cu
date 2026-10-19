#include<iostream>
#include<bits/stdc++.h>
#include<cuda.h> 
#include<cuda_runtime.h>
#include<ECLgraph.h>

using namespace std; 
typedef unsigned long long ull; 
const unsigned long long INF = ~0ULL;



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



/**
 * @brief Check the calls 
 * @note inherited from the ECL-MST code. 
 * @param line 
 */
static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


/**
 * @brief This method will return the parent of the node v
 * 		  Read only implementation with no explicit path contraction 
 * 
 * @todo add parallel safe implementation
 * @param parents 
 * @param src 
 * @return __host__ 
 */
__device__ int device_find( const int* parents, const int v){
	return 0; 
}




/**
 * @brief Initialize the parents array allocated on the GPU 
 * 		  initially every node is parent of itself. 
 * 
 * @param parents 
 * @param N 
 * @return __global__ 
 */
__global__ void init_parents( int *parents, int N ){
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	if(gid >= N ) return ; 
	parents[gid] = gid; 
}




/**
 * @brief 
 * 	
 * @param G ELCgraph containing data on host 
 * @param inMST boolean vector to know if edge is in MST 
 */
void GPU_MST(ECLgraph G, bool*h_MST 	){

	bool* d_MST = NULL; 
	cudaMalloc(&d_MST, G.edges*sizeof(bool)); 

	// parent array for which works as DSU here, device only for now 
	int* d_parent = NULL; 
	cudaMalloc(&d_parent, G.nodes*sizeof(int)); 

	// d_cheapsest array to hold cheapest outgoing edge from the node i, encoded edge hence datatype will be unsigned long long 
	ull* d_cheapest = NULL; 
	cudaMalloc(&d_cheapest, G.nodes*sizeof(ull)); 

	// Allocate memory for the ECL_Graph array on GPU
	int* d_nindex = NULL, *d_nlist = NULL, *d_eweight = NULL ; 
	cudaMalloc(&d_nindex, G.nodes*sizeof(int) + 1 ); 
	cudaMalloc(&d_nlist, G.edges*sizeof(int)); 
	cudaMalloc(&d_eweight, G.edges*sizeof(int)); 

	// copy data from ECL_graph G to device pointers
	cudaMemcpy(d_nindex,G.nindex,G.nodes*sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_nlist, G.nlist, G.edges*sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_eweight, G.eweight, G.edges*sizeof(int), cudaMemcpyHostToDevice);

	// now data is ready on GPU we can start the MST computation loop, 
	// varaible to decide when to terminate the algorithsm 
	bool h_changed = true; 
	bool* d_changed = NULL; 
	cudaMalloc(&d_changed, sizeof(bool)); 
	cudaMemset(d_changed, 1, sizeof(bool)); // set as true before starting of roud, at loop begining set to false again. 

	// some basic kernels to initialize the d_parent and d_cheapest array
	// parent array initialization is just one time. but d_cheapest needs to reset after every round 
	int TPB = 256; 
	int Blocks = (G.nodes + TPB - 1 ) / TPB; 
	init_parents<<<Blocks, TPB>>>(d_parent, G.nodes);
	

	while( h_changed ){
		
		// set h_changed and d_changed to false; 
		h_changed = false; 
		cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice); 

		// reset the cheapest array; 
		cudaMemset(d_cheapest,0xFF,G.nodes*sizeof(ull)); 
		// Find the cheapest outgoing edge from each component and store it's edge index in cheapest[comp]; 


		// write here kernle to construct the MST and unify the components using the cheapest outgoing edge

		// compress tree kernel -> flattens the whole tree(parents array); 

		// get the value of d_changed 
		cudaMemcpy(&h_changed, d_changed, sizeof(bool),cudaMemcpyDeviceToHost);
	}
	


}

int main(){
	
	return 0 ; 
} 

