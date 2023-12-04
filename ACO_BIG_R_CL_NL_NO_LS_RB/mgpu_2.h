/*
* Author: Juan Manuel Aedo Sepulveda
* This is an Ant Colony Algorithm
*/
#ifndef MGPU_H_
#define MGPU_H_
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <float.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#define N_e 30
#define ITERACION 3000
#define M 1280    //a partir de 512 si duplico 
#define n_new_edge 8
#define Q 1 // 
#define n_best 10
#define N 24104
#define LS_ITERATION 0
#define c_l 80
#define cl 80
#define name_e "problems/xrh24104.tsp"
#define name_test_1 "iteration_time/interation_time_xrh24104.csv"
#define name_test_2 "iteration_time_series/interation_time_xrh24104.csv"
#define name_test_3 "warm_up_time/warm_up_time_xrh24104.txt"
#define name_test_4 "soluciones/soluciones_xrh24104.csv"
#define solucion 426 //eil51
//mona-lisa100K
//#define solucion 7542.0 //berlin52
//#define solucion 675.0 //st70

//#define solucion 538.0 //eil76
//#define solucion 1211.0 //rat99
//#define solucion 21282.0 //kroA100
//#define solucion 629.0 //eil101
//#define solucion 6942.0 //gr120
//#define solucion 6528.0 //ch150
//#define solucion 29368.0 //kroA200
//#define solucion 50778.0 //pcb442
//#define solucion 2763.0 //pa561
// fnl4461
void lectura_2(double *dis);
//-------------------------------------------------------------------------------------
void guardar_warm_up(double *time,double alpha,double beta,double e);
void guardar_iteration_time(double *time,double alpha,double beta,double e);
void guardar_soluciones(int *soluciones,double alpha,double beta,double e);
void guardar_iteration_time_series(double *time,double alpha,double beta,double e);
float std_vec(float *vec, float prom);
float std_vec_it(int *vec,float prom);
float minimovec(float *vec);

int EUC_2D_C(double *d_d,int p1,int p2);
int rutainicial(int *rute_op,double *d,bool *lista_vis);
/*
*This function free the memory for the given vectores
*/

/*
* This function join the 4 gpu cost vectors
*/
__global__ void LIST_INIT(int *NEW_LIST,int *d_NEW_LIST_INDX);
__global__ void PHEROMONE_UPDATE_MMAS(int *ROUTE,int *BEST_ANT,
double *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,
int BEST_GLOBAL_SOLUTION);
__global__ void PHEROMONE_CHECK_MMAS(double *PHEROMONE_MATRIX,double tau_max,double tau_mim);
/*
* This function calculate the combinated factor of pheromone and visibility
*/
/*
* This function reset the rutes to 0
*/
__global__ void RESET_ROUTE(int *ROUTE);
/*
* This function join the 4 gpu rute vectors
*/
/*
* This function set the pheromone to a value 
*/
__global__ void fijar_pheromone(double *d_pheromone,double ini_pheromone);
/*
* This function set the visibility to a value 
*/
/*
* This function set the cost to a value 
*/
/*
* This function reset the visited_list of the ant tour
*/
__global__ void HEURISTIC_PHEROMONE_CALCULATION(double *NODE_COORDINATE,double *PHEROMONE_MATRIX,double *HEURISTIC_PHEROMONE,int *NN_LIST,double alpha,double beta);
__global__ void RESET_VISITED_LIST(bool *VISITED_LIST);
__global__ void EVAPORATION(double *PHEROMONE_MATRIX,double e);
__global__ void ANT_SOLUTION_CONSTRUCT(double *HEURISTIC_PHEROMONE,double *NODE_COORDINATE_2D,int di,
int *PREDECESSOR_ROUTE,int *SUCCESSOR_ROUTE,
int max_new_edges,curandState *state,int *NN_LIST,int *NEW_LIST,int *NEW_LIST_INDX);
__global__ void ANT_COST_CALCULATION_LS(int *ROUTE,int *COST,double *NODE_COORDINATE_2D,int *ROUTE_AUX,curandState *state);
__global__ void PHEROMONE_UPDATE(int *ROUTE,int *BEST_ANT,double *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION);
__global__ void iniciar_kernel(curandState *state,int di);
__device__ void L_S(double *d_d,int *d_rute,double *d_global_cost,int *local_search_list,
int *d_rute_succ_all,int *d_rute_pred_all,int *d_orden_indice,int i);
__device__ void opt3(int *ROUTE,double *NODE_COORDINATE_2D,int *COST,int k,int *ROUTE_AUX,curandState *state);
__device__ int EUC_2D(double *d_d,int x,int y);
__device__ void CHECK_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int CHOSEN_NODE);
__device__ bool IS_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j);
__device__ int GET_CANDIDATE(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j);
#endif
