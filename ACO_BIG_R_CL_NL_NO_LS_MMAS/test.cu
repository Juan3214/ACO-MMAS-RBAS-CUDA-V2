#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <float.h>
#include <thrust/device_ptr.h>
#include <omp.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include "mgpu_2.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
int main(){
    printf("\n hola mundo \n");
    
    cudaSetDevice(0);
    double *NODE_COORDINATE_2D;NODE_COORDINATE_2D=(double*)calloc(N*2,sizeof(double));
    int *DISTANCE_NODE;DISTANCE_NODE=(int*)calloc(N,sizeof(int));
    int *d_DISTANCE_NODE;cudaMalloc( (void **) &d_DISTANCE_NODE, (N)*sizeof( int ));
    int *d_NN_LIST_aux;cudaMalloc( (void **) &d_NN_LIST_aux, (N)*sizeof( int ));
    int *d_NN_LIST_cl;cudaMalloc( (void **) &d_NN_LIST_cl, (N*cl)*sizeof( int ));
    int *NN_LIST_cl;NN_LIST_cl=(int*)calloc(N*c_l,sizeof(int));
    lectura_2(NODE_COORDINATE_2D);
    printf("\n termine de leer");
    int i,j;
    thrust::device_ptr<int> dev_inx_d = thrust::device_pointer_cast(d_NN_LIST_aux);
    thrust::device_ptr<int> dev_d = thrust::device_pointer_cast(d_DISTANCE_NODE);
    for (i=0;i<N;i++){
        if (i%1000==0){
            printf("%d \n",i);
        }
        for (j=0;j<N;j++){
            DISTANCE_NODE[j]= EUC_2D_C(NODE_COORDINATE_2D,i,j); 
        }
        cudaMemcpy(d_DISTANCE_NODE,DISTANCE_NODE,N*sizeof(int),cudaMemcpyHostToDevice);
        thrust::sequence(thrust::device,dev_inx_d, dev_inx_d+N); // se ordenan de menor distancia a mayor distancia
        thrust::sort_by_key(thrust::device,dev_d, dev_d +N, dev_inx_d,thrust::less<int>());
        cudaDeviceSynchronize();
        cudaMemcpy(NN_LIST_cl+i*cl,d_NN_LIST_aux+1,cl*sizeof(int),cudaMemcpyDeviceToHost);
        
        //cudaMemcpy(NN_LIST_cl,d_NN_LIST_aux+1,cl*N*sizeof(int),cudaMemcpyDeviceToHost);
        
    }
    printf("\n termine de ordenar");
    cudaMemcpy(d_NN_LIST_cl,NN_LIST_cl,N*cl*sizeof(int),cudaMemcpyHostToDevice);
    int *vec_soultion;double prom_time_2=0;int *vec_iter;int exito=0;double *vec_warm_up_time;
    vec_soultion= (int* )malloc(N_e*sizeof(int));vec_iter= (int* )malloc(N_e*sizeof(int)); //vectores para estadistica
    double *vec_iteration_time;vec_iteration_time=(double*)malloc(N_e*sizeof(double));
    memset(vec_soultion,0,N_e*sizeof(int));vec_warm_up_time= (double* )malloc(N_e*sizeof(double));
    double *vec_ant_iteration_time_series;vec_ant_iteration_time_series=(double*)malloc(ITERACION*sizeof(double));
    for (i=0;i<ITERACION;i++)vec_ant_iteration_time_series[i]=0.0;
    int x;
    float elapsed_for_gpu_ant[4];
    double alpha=0.3;
    double beta=5;
    double e=0.01;
    for (x=0;x<N_e;x++){
    cudaEvent_t start_events[4];
    cudaEvent_t end_events[4];
    double prom_time=0;
    double tau_mim,tau_max;
    double P_best=0.001,avg=(double)N/2.0;
    //int *d_PREDECESSOR_ROUTE_MGPU[4],*d_SUCCESSOR_ROUTE_MGPU[4]; //2 SOLO PARA FOCUSED
    int *d_COST_MGPU[4];int *d_NN_LIST_CL_MGPU[4]; //4
    double *d_NODE_COORDINATE_MGPU[4];  // 5
    int *d_PREDECESSOR_ROUTE_OP_MGPU[4],*d_SUCCESSOR_ROUTE_OP_MGPU[4];  //8
     //9
    int *d_NEW_LIST[4],*d_NEW_LIST_INDX[4],*d_ROUTE_AUX[4];
    double *d_HEURISTIC_PHEROMONE_MGPU[4]; //10
    /* 10 MULTI GPU VARIABLES */
    int *NEW_LIST_GLOBAL,*NEW_LIST_INDX_GLOBAL;
    NEW_LIST_GLOBAL=(int*)malloc(M*4*(N+1)*sizeof(int));
    NEW_LIST_INDX_GLOBAL=(int*)malloc(M*4*(N+1)*sizeof(int));
    int *d_BEST_ANT,*d_OPTIMAL_ROUTE,*d_GLOBAL_COST,*d_GLOBAL_NEW_LIST; //4  
    int *d_ROUTE_NN;
    cudaMalloc( (void **) &d_ROUTE_NN, N_e*(N+1)*sizeof(int) );//1
    cudaSetDevice(0);
    double *d_HEURISTIC_PHEROMONE,*d_PHEROMONE_MATRIX; //6
    /*6 MASTER GPU VARIABLES*/
    double *HEURISTIC_PHEROMONE;
    bool *VISITED_LIST;
    int *GLOBAL_ROUTE,*OPTIMAL_ROUTE,*GLOBAL_COST,*BEST_ANT;
    int *PREDECESSOR_ROUTE, *SUCCESSOR_ROUTE;
    double *d_NODE_COORDINATE_2D;
    curandState *d_state[4];
    // global route x2 
    printf("\n fijando memoria en gpu \n");
    cudaMalloc( (void **) &d_BEST_ANT, 4*M*sizeof(int) );//1
    cudaMalloc( (void **) &d_OPTIMAL_ROUTE, (N+1)*sizeof(int) );//2
    cudaMalloc( (void **) &d_HEURISTIC_PHEROMONE, N*cl*sizeof(double));//3
    cudaMalloc( (void **) &d_PHEROMONE_MATRIX, N*cl*sizeof(double) );//4
    cudaMalloc( (void **) &d_GLOBAL_COST, 4*M*sizeof( int ) );//5
    cudaMalloc( (void **) &d_GLOBAL_NEW_LIST, 4*(N+1)*M*sizeof( int ) );//6

    cudaMalloc( (void **) &d_NODE_COORDINATE_2D, N*2*sizeof(double ) );//6
    cudaMemcpy(d_NODE_COORDINATE_2D,NODE_COORDINATE_2D,2*N*sizeof( double ),cudaMemcpyHostToDevice);
    printf("\n fijando memoria en cpu \n");

    OPTIMAL_ROUTE=(int*)malloc((N+1)*sizeof(int));
    PREDECESSOR_ROUTE=(int*)malloc(N*sizeof(int));
    SUCCESSOR_ROUTE=(int*)malloc(N*sizeof(int));
    GLOBAL_COST=(int*)malloc(4*M*sizeof(int));
    HEURISTIC_PHEROMONE=(double*)malloc(N*cl*sizeof(double));
    //LOCAL_SEARCH_LIST_MGPU=(int*)malloc(4*M*N*sizeof(int));
    VISITED_LIST=(bool*)malloc(N*sizeof(bool));
    BEST_ANT=(int*)malloc(4*M*sizeof(int));
    for (i=0;i<N;i++)VISITED_LIST[i]=false; 
    //12
    printf("\n fijando memoria en mgpu \n");
    for (i=0;i<4;i++){
        cudaSetDevice(i);
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
        cudaMalloc( (void **) &d_NODE_COORDINATE_MGPU[i], 2*N*sizeof( double ) );//1
        cudaMemcpy(d_NODE_COORDINATE_MGPU[i],NODE_COORDINATE_2D,2*N*sizeof( double ),cudaMemcpyHostToDevice);
        cudaMalloc( (void **) &d_NEW_LIST[i], M*(N+1)*sizeof( int ) );//2
        cudaMalloc( (void **) &d_ROUTE_AUX[i], M*(N+1)*sizeof( int ) );//2
        cudaMalloc( (void **) &d_NEW_LIST_INDX[i], M*(N+1)*sizeof( int ) );
        //cudaMalloc( (void **) &d_PREDECESSOR_ROUTE_MGPU[i], M*N*sizeof( int ) );//4
        //cudaMalloc( (void **) &d_SUCCESSOR_ROUTE_MGPU[i], M*N*sizeof( int ) );//5
        cudaMalloc( (void **) &d_COST_MGPU[i], M*sizeof( int ) );//6
        //cudaMalloc( (void **) &d_LOCAL_SEARCH_LIST_MGPU[i], M*N*sizeof( int ) );
        cudaMalloc( (void **) &d_NN_LIST_CL_MGPU[i], cl*N*sizeof( int ) ); //7
        cudaMemcpy(d_NN_LIST_CL_MGPU[i],NN_LIST_cl,(c_l*N)*sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc( (void **) &d_PREDECESSOR_ROUTE_OP_MGPU[i], N*sizeof( int ) );//8
        cudaMalloc( (void **) &d_SUCCESSOR_ROUTE_OP_MGPU[i], N*sizeof( int ) );//9
        cudaMalloc( (void **) &d_HEURISTIC_PHEROMONE_MGPU[i], N*cl*sizeof( double ) );//10
    }
    cudaSetDevice(0);
    printf("\n greedy \n");
    int BEST_GLOBAL_SOLUTION=rutainicial(OPTIMAL_ROUTE,NODE_COORDINATE_2D,VISITED_LIST);
    cudaMemcpy(d_OPTIMAL_ROUTE,OPTIMAL_ROUTE, (N+1)*sizeof(int),cudaMemcpyHostToDevice);

    double ini_pheromone=(double)BEST_GLOBAL_SOLUTION;
    double p=pow(P_best,1/(double)N);
    
    printf("\n fijando feromona %lf \n",(double)1/ini_pheromone);
    fijar_pheromone<<<(N*cl+32-(N*cl%32)),32>>>(d_PHEROMONE_MATRIX,(double)1/ini_pheromone);

    HEURISTIC_PHEROMONE_CALCULATION<<<N,cl>>>(d_NODE_COORDINATE_2D,d_PHEROMONE_MATRIX,
    d_HEURISTIC_PHEROMONE,d_NN_LIST_cl,alpha,beta);


    thrust::device_ptr<int> dev_inx = thrust::device_pointer_cast(d_BEST_ANT); //utilziar thrust para sorting
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_GLOBAL_COST);      
    
    
    
    cudaMemcpy(HEURISTIC_PHEROMONE,d_HEURISTIC_PHEROMONE,(N*c_l)*sizeof(double),cudaMemcpyDeviceToHost);
    printf("\n copiando predecesor\n");


    
    for (i=0;i<N;i++){
        PREDECESSOR_ROUTE[OPTIMAL_ROUTE[i]]=OPTIMAL_ROUTE[i+1]; //seguarda el eje (u,v) osea si tengo rute_predecessor[k]=j quiere decir que en la ruta optima la ciudad k esta conectada con j
        if (i!=0){
            SUCCESSOR_ROUTE[OPTIMAL_ROUTE[i]]=OPTIMAL_ROUTE[i-1];
        }
        else{
            SUCCESSOR_ROUTE[OPTIMAL_ROUTE[i]]=OPTIMAL_ROUTE[N-1];
        }
    }
    for (i=0;i<4;i++){
        cudaSetDevice(i);
        cudaMemcpy(d_PREDECESSOR_ROUTE_OP_MGPU[i],PREDECESSOR_ROUTE,(N)*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_SUCCESSOR_ROUTE_OP_MGPU[i],SUCCESSOR_ROUTE,(N)*sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc((void **) &d_state[i], M*sizeof(curandState));iniciar_kernel<<<32,M/32>>>(d_state[i],i);
        cudaMemcpy(d_HEURISTIC_PHEROMONE_MGPU[i],HEURISTIC_PHEROMONE,N*cl*sizeof(double),cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    int mejor,it,LAST_IMPROVE_ITERATION=0;
    // ITERACIONES 

    int num_gpus;
    size_t freee, total;
    cudaGetDeviceCount( &num_gpus );
    
    cudaSetDevice(0);
    printf("\n INICIANDO ITERACIONES\n");
    for (it=0;it<ITERACION;it++){
        double begin_1 =omp_get_wtime();
        #pragma omp parallel for num_threads(4)
        for (i=0;i<4;i++){
            cudaSetDevice(i);
            cudaEventRecord(start_events[i]);
            LIST_INIT<<<N,min(M,1024)>>>(d_NEW_LIST[i],d_NEW_LIST_INDX[i]);
            ANT_SOLUTION_CONSTRUCT<<<M/4,4>>>(d_HEURISTIC_PHEROMONE_MGPU[i],d_NODE_COORDINATE_MGPU[i],
           i,d_PREDECESSOR_ROUTE_OP_MGPU[i],
            d_SUCCESSOR_ROUTE_OP_MGPU[i],8,d_state[i],d_NN_LIST_CL_MGPU[i],d_NEW_LIST[i],d_NEW_LIST_INDX[i]);
            /*------------------------ SOBRE ANT_SOLUTION_CONSTRUCT--------------------*/            
            //aumentar el numero de thread ahora parece mejorar el rendimiento
            //en el alg anterior no ocurria eso, cuidado con la memoria compartida
            //por ahora con 4 threads y 4 bloques es lo optimo
            /*------------------------ SOBRE ANT_COST_CALCULATION_LS --------------------*/
            // aumentar el numero de threads mejora el rendimiento  hasta 32 thread
            ANT_COST_CALCULATION_LS<<<M/32,32>>>(d_NEW_LIST[i],d_COST_MGPU[i],d_NODE_COORDINATE_MGPU[i],d_ROUTE_AUX[i],d_state[i]);
            gpuErrchk(cudaMemcpyAsync(GLOBAL_COST+i*M,d_COST_MGPU[i],M*sizeof(int),cudaMemcpyDeviceToHost));   //esto ahorra 6 ms en 4000 nodos 
            gpuErrchk(cudaMemcpyAsync(NEW_LIST_GLOBAL+i*M*(N+1),d_NEW_LIST[i],(N+1)*M*sizeof(int),cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpyAsync(NEW_LIST_INDX_GLOBAL+i*M*(N+1),d_NEW_LIST_INDX[i],(N+1)*M*sizeof(int),cudaMemcpyDeviceToHost));   //esto ahorra 6 ms en 4000 nodos 
            cudaDeviceSynchronize();

            cudaEventRecord(end_events[i]);

        }
        cudaDeviceSynchronize();
        double end_1 =omp_get_wtime();
        /*
        for (i=0;i<N;i++){
            printf("%d ",NEW_LIST_GLOBAL[63*(N+1)+N-1-i]-GLOBAL_ROUTE[63*(N+1)+i]);
        }*/
        if(it==0)vec_warm_up_time[x]=(end_1-begin_1)*1000;
        vec_ant_iteration_time_series[it]+=((end_1-begin_1)*1000.0)/((double)N_e);
        
        //printf("\n termino el recorrido en %lf ms\n",(end_1-begin_1)*1000);
        /*
        for(i = 0; i < 4; i++)
            {
                //the end_event must have happened to get a valid duration
                //In this example, this is true because of previous device synchronization
                cudaEventElapsedTime(&elapsed_for_gpu_ant[i], start_events[i], end_events[i]);
                printf("Elapsed time on device %d: %f ms\n", i, elapsed_for_gpu_ant[i]);
            }
        */
        //for (i=0;i<4;i++){
          //  cudaSetDevice(i);
            
        //}
        
        cudaSetDevice(0);
        gpuErrchk(cudaMemcpy(d_GLOBAL_COST,GLOBAL_COST,4*M*sizeof(int),cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_GLOBAL_NEW_LIST,NEW_LIST_GLOBAL,4*(N+1)*M*sizeof(int),cudaMemcpyHostToDevice));
        thrust::sequence(thrust::device,dev_inx, dev_inx+4*M);
        thrust::sort_by_key(thrust::device,dev_ptr, dev_ptr + 4*M, dev_inx,thrust::less<int>());
        cudaMemcpy(GLOBAL_COST,d_GLOBAL_COST,4*M*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(BEST_ANT,d_BEST_ANT,4*M*sizeof(int),cudaMemcpyDeviceToHost);
                
        
        mejor=BEST_ANT[0];  
        cudaSetDevice(0);
        if (it==0){
            GLOBAL_COST[0]=BEST_GLOBAL_SOLUTION;
            tau_max=(double)e*((double)1/(double)BEST_GLOBAL_SOLUTION);
            tau_mim=tau_max*((1-p)/((avg-1)*p));
        }
        if (it%100==0)printf("\n %d\n",GLOBAL_COST[0]);
        if (GLOBAL_COST[0]<BEST_GLOBAL_SOLUTION){
                BEST_GLOBAL_SOLUTION=GLOBAL_COST[0];
                LAST_IMPROVE_ITERATION=it;
                tau_max=(double)e*((double)1/(double)BEST_GLOBAL_SOLUTION);
                tau_mim=tau_max*((1-p)/((avg-1)*p));
                if (tau_max<tau_mim)tau_mim=tau_max;
                printf("\n mejor global = %d en iter= %d en experimento %d\n",BEST_GLOBAL_SOLUTION,LAST_IMPROVE_ITERATION,x);
                //printf("\n tau max =%.16lf \n tau min =%.16lf \n",tau_max,tau_mim);
                for (i=0;i<N+1;i++){
                    OPTIMAL_ROUTE[i]=NEW_LIST_GLOBAL[mejor*(N+1)+i%N];
                    //printf("%d ",OPTIMAL_ROUTE[i]);
                    if (i>0){
                        PREDECESSOR_ROUTE[OPTIMAL_ROUTE[i-1]]=OPTIMAL_ROUTE[i];
                    }
                    if (i!=0){
                        SUCCESSOR_ROUTE[OPTIMAL_ROUTE[i]]=OPTIMAL_ROUTE[i-1];
                    }
                    else{
                        SUCCESSOR_ROUTE[OPTIMAL_ROUTE[i]]=OPTIMAL_ROUTE[N-1];
                    }
                }
                cudaMemcpy(d_OPTIMAL_ROUTE,OPTIMAL_ROUTE, (N+1)*sizeof(int),cudaMemcpyHostToDevice);
                for (i=0;i<4;i++){
                    cudaSetDevice(i);
                    cudaMemcpy(d_SUCCESSOR_ROUTE_OP_MGPU[i],SUCCESSOR_ROUTE, N*sizeof(int),cudaMemcpyHostToDevice);
                    cudaMemcpy(d_PREDECESSOR_ROUTE_OP_MGPU[i],PREDECESSOR_ROUTE, N*sizeof(int),cudaMemcpyHostToDevice);
                }
                cudaSetDevice(0);
        }
        cudaSetDevice(0);
        EVAPORATION<<<((N*cl+32-(N*cl%32)))/32,32>>>(d_PHEROMONE_MATRIX,e);
        /*-----------------------RANK BASED--------------------------------*/
        //PHEROMONE_UPDATE<<<((N+32-(N%32)))/32,32>>>(d_GLOBAL_NEW_LIST,d_BEST_ANT,d_PHEROMONE_MATRIX,
        //d_NN_LIST_cl,d_GLOBAL_COST,d_OPTIMAL_ROUTE,BEST_GLOBAL_SOLUTION);
        /*-----------------------MMAS      --------------------------------*/
        PHEROMONE_CHECK_MMAS<<<((N*cl+32-(N*cl%32)))/32,32>>>(d_PHEROMONE_MATRIX, tau_max, tau_mim);
        //PHEROMONE_UPDATE<<<((N+32-(N%32)))/32,32>>>(d_GLOBAL_NEW_LIST,d_BEST_ANT,d_PHEROMONE_MATRIX,
        //d_NN_LIST_cl,d_GLOBAL_COST,d_OPTIMAL_ROUTE,BEST_GLOBAL_SOLUTION);
        PHEROMONE_UPDATE_MMAS<<<((N+32-(N%32)))/32,32>>>(d_GLOBAL_NEW_LIST,d_BEST_ANT,d_PHEROMONE_MATRIX,
        d_NN_LIST_cl,d_GLOBAL_COST,d_OPTIMAL_ROUTE,BEST_GLOBAL_SOLUTION);
        PHEROMONE_CHECK_MMAS<<<((N*cl+32-(N*cl%32)))/32,32>>>(d_PHEROMONE_MATRIX, tau_max, tau_mim);
        cudaMemcpy(HEURISTIC_PHEROMONE,d_PHEROMONE_MATRIX,cl*N*sizeof(double),cudaMemcpyDeviceToHost);        
        
        HEURISTIC_PHEROMONE_CALCULATION<<<N,cl>>>(d_NODE_COORDINATE_2D,d_PHEROMONE_MATRIX,
        d_HEURISTIC_PHEROMONE,d_NN_LIST_cl,alpha,beta);
        
        
        cudaMemcpy(HEURISTIC_PHEROMONE,d_HEURISTIC_PHEROMONE,(N*c_l)*sizeof(double),cudaMemcpyDeviceToHost);
        /*
        for (i=0;i<N;i++){
            printf("\n");
            for (j=0;j<cl;j++)printf("\n %.16lf",HEURISTIC_PHEROMONE[i*cl+j]);
        }*/
        for (i=0;i<4;i++){
            cudaSetDevice(i);
            cudaMemcpy(d_HEURISTIC_PHEROMONE_MGPU[i],HEURISTIC_PHEROMONE,(N*c_l)*sizeof(double),cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        double end_2 =omp_get_wtime(); 
        if(it!=0)prom_time+=(end_2-begin_1)*1000;
    }
    vec_soultion[x]=BEST_GLOBAL_SOLUTION;
    printf("\n -------------------------------\n");
    //for (i=0;i<N+1;i++)printf("%d ", OPTIMAL_ROUTE[i]);
    printf("\n -------------------------------\n");
    cudaSetDevice(0);
    cudaFree(d_ROUTE_NN);free(BEST_ANT);cudaFree(d_GLOBAL_NEW_LIST);
    cudaFree(d_NODE_COORDINATE_2D);
    cudaFree(d_BEST_ANT);cudaFree(d_OPTIMAL_ROUTE);cudaFree(d_GLOBAL_COST);
    cudaFree(d_HEURISTIC_PHEROMONE);cudaFree(d_PHEROMONE_MATRIX);
    for (i=0;i<4;i++){
        cudaSetDevice(i);cudaFree(d_NODE_COORDINATE_MGPU[i]);
        //cudaFree(d_PREDECESSOR_ROUTE_MGPU[i]);
        //cudaFree(d_SUCCESSOR_ROUTE_MGPU[i]);
        cudaFree(d_ROUTE_AUX[i]);
        cudaFree(d_COST_MGPU[i]);
        cudaFree(d_state[i]);
        cudaFree(d_NEW_LIST[i]);
        cudaFree(d_NEW_LIST_INDX[i]);
        //cudaFree(d_LOCAL_SEARCH_LIST_MGPU[i]);
        cudaFree(d_NN_LIST_CL_MGPU[i]);
        cudaFree(d_PREDECESSOR_ROUTE_OP_MGPU[i]);
        cudaFree(d_SUCCESSOR_ROUTE_OP_MGPU[i]);
        cudaFree(d_HEURISTIC_PHEROMONE_MGPU[i]);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(end_events[i]);
    }
    //free(LOCAL_SEARCH_LIST_MGPU);
    free(OPTIMAL_ROUTE);free(VISITED_LIST);
    free(SUCCESSOR_ROUTE);free(PREDECESSOR_ROUTE);free(GLOBAL_COST);
    free(HEURISTIC_PHEROMONE);free(NEW_LIST_GLOBAL);free(NEW_LIST_INDX_GLOBAL);
    prom_time/=(ITERACION);
    prom_time_2+=prom_time;
    vec_iteration_time[x]=prom_time;
    }
    
    prom_time_2/=(N_e);

    printf("\n el tiempo promedio es de %lf\n ",prom_time_2);
    guardar_warm_up(vec_warm_up_time,alpha,beta,e);guardar_soluciones(vec_soultion,alpha,beta,e);
    guardar_iteration_time_series(vec_ant_iteration_time_series,alpha,beta,e);
    guardar_iteration_time(vec_iteration_time,alpha,beta,e);
    free(NODE_COORDINATE_2D);free(DISTANCE_NODE);
    cudaFree(d_DISTANCE_NODE);free(NN_LIST_cl);cudaFree(d_NN_LIST_aux);cudaFree(d_NN_LIST_cl);
    
    
    
    
    return 0;
}
