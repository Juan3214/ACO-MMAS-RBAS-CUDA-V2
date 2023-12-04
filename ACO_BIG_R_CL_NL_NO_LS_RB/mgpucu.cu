#include "mgpu_2.h"
//no necesito list local search particular por gpu . borrar
// LOCAL SEARCH LIST GUARDA POSICION EN LA RUTA, NO VERTICES
__global__ void iniciar_kernel(curandState *state,int di){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    curand_init((unsigned long long)clock() + i+M*di, 0, 0, &state[i]);
} 
__global__ void ANT_SOLUTION_CONSTRUCT(double *HEURISTIC_PHEROMONE,double *NODE_COORDINATE_2D,
int di,int *PREDECESSOR_ROUTE,int *SUCCESSOR_ROUTE,
int max_new_edges,curandState *state,int *NN_LIST,int *NEW_LIST,int *NEW_LIST_INDX){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    int id=threadIdx.x;
    if (i<M){
        int loc_act;int j=0;int new_edges=0;
        curandState localstate=state[i];
        float myrandf = curand_uniform(&localstate);
        myrandf *= (N-2 + 0.999999);
        myrandf += 0;
        int random = (int)truncf(myrandf);
       
        CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,random);
        int pos;  
        int r;

        __shared__ double prob[(c_l)*4];
        while(j<N-1){
            int kas=0;
            loc_act=NEW_LIST[i*(N+1)+NEW_LIST[i*(N+1)+N]]; //locacion actual
             //se marca como visitada
            prob[id*cl]=HEURISTIC_PHEROMONE[loc_act*cl]*(1-(int)(IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,NN_LIST[loc_act*cl]))); //si fue visitada la probabilidad es 0
            for (int k=1;k<cl;k++){
                    pos=NN_LIST[loc_act*cl+k];
                    prob[id*cl+k]=prob[id*cl+k-1]+HEURISTIC_PHEROMONE[loc_act*cl+k]*(1-(int)(IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,pos)));
            }
            double val =  curand_uniform_double(&localstate);
            double ranval=val*prob[id*cl+cl-1];
            int CHOSEN_NODE=-9;
            for (int ran=0;ran<cl;ran++){
                if (ranval<=prob[id*cl+ran]){
                    CHOSEN_NODE=NN_LIST[loc_act*cl+ran]; //revisar indices
                    kas=1;// Quesooo, atento cuando se cumple esta condicion
                    break;
                }
            }
            if (ranval==0.0){
                int min=INT_MAX;int dist; 
                for (int k=0;k<NEW_LIST[i*(N+1)+N];k++){
                    int candidate=GET_CANDIDATE(NEW_LIST,NEW_LIST_INDX,i,k);
                    dist=EUC_2D(NODE_COORDINATE_2D,candidate,loc_act);
                    if (IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,candidate)==false && dist<min ){
                        CHOSEN_NODE=candidate; //revisar indices
                        min=dist;
                    }
                }
            }
            CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,CHOSEN_NODE);
            if (CHOSEN_NODE==loc_act){
                for(int r=0;r<1000;r++)printf("\n ERROR \n");
            }
            
            j++;
            //desde aqui es mas codigo del polaco
            //se rellena con la ruta original
            if (SUCCESSOR_ROUTE[loc_act]!=CHOSEN_NODE){
                new_edges+=1;
                //local_search_list[i*(N+1)+new_edges]=j;
            }
            int u;
            if (new_edges>=max_new_edges){
                u=SUCCESSOR_ROUTE[CHOSEN_NODE];
                //if(i==0)printf("\n new edges %d \n",new_edges);
                while (IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,u)==false && j<N-1){
                    
                    
                    CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,u);
                    u=SUCCESSOR_ROUTE[u];
                    
                    j++; 
                    //printf("\n yes yes");
                }
                u=PREDECESSOR_ROUTE[u];
                //if (i==0)printf("\n %d y j %d",d_rute_pred[u],j);
                
                while (IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,u)==false && j<N-1){ 
                    CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,u);
                    u=PREDECESSOR_ROUTE[u];
                    //printf("\n yes no");
                    j++;
                }    
            }
        }
        //local_search_list[i*(N+1)]=new_edges;
        state[i]=localstate;
    }
}
__device__ void CHECK_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int CHOSEN_NODE){
    int length=NEW_LIST[ant*(N+1)+N]-1;
    int CHOSEN_INDX=NEW_LIST_INDX[ant*(N+1)+CHOSEN_NODE];
    int temp_1=NEW_LIST[ant*(N+1)+length];
    int temp_2=NEW_LIST_INDX[ant*(N+1)+temp_1]  ;
    NEW_LIST[ant*(N+1)+length]=CHOSEN_NODE;
    NEW_LIST_INDX[ant*(N+1)+temp_1]=CHOSEN_INDX;
    NEW_LIST[ant*(N+1)+CHOSEN_INDX]=temp_1;
    NEW_LIST_INDX[ant*(N+1)+CHOSEN_NODE]=temp_2;
    NEW_LIST[ant*(N+1)+N]-=1;
}

__device__ bool IS_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j){
    if (NEW_LIST_INDX[ant*(N+1)+j]<NEW_LIST[ant*(N+1)+N]){
        return false;
    }
    else{
        return true;
    }
}
__device__ int GET_CANDIDATE(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j){
    return NEW_LIST[ant*(N+1)+j];
}
__global__ void LIST_INIT(int *d_NEW_LIST,int *d_NEW_LIST_INDX){
     //estas son las hormigas
    int j=blockIdx.x; //estas son las ciudades
    if (j<N){
        for (int i=threadIdx.x;i<M;i+=blockDim.x){
        if (j==0){
            d_NEW_LIST[i*(N+1)+N]=N;
            d_NEW_LIST_INDX[i*(N+1)+N]=N;
        }
        d_NEW_LIST[i*(N+1)+j]=j;
        d_NEW_LIST_INDX[i*(N+1)+j]=j;
    }
    }
}
__global__ void ANT_COST_CALCULATION_LS(int *ROUTE,int *COST,double *NODE_COORDINATE_2D,int *ROUTE_AUX,curandState *state){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if( i<M){
        int j;
        int SUMA=0;
        for (j=0;j<N;j++){
            SUMA+=EUC_2D(NODE_COORDINATE_2D,ROUTE[i*(N+1)+j],ROUTE[i*(N+1)+(j+1)%N]);
        }
        COST[i]=SUMA;
        for (j=0;j<LS_ITERATION;j++)opt3(ROUTE,NODE_COORDINATE_2D,COST,i,ROUTE_AUX,state);
    }
}
__global__ void PHEROMONE_UPDATE(int *ROUTE,int *BEST_ANT,double *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION){
    int j=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (j<N){
    int ant,k;
    for (ant=0;ant<n_best-1;ant++)
        for (k=0;k<cl;k++){
            //printf("\n Wow %d \n",ROUTE[BEST_ANT[ant]*(N+1)+j+1]);
            if (ROUTE[BEST_ANT[ant]*(N+1)+(j+1)%N]==NN_LIST[ROUTE[BEST_ANT[ant]*(N+1)+j%N]*cl+k]){
                PHEROMONE_MATRIX[ROUTE[BEST_ANT[ant]*(N+1)+j%N]*cl+k]+=(1.0*(n_best-1-ant))/((double)COST[ant]);
                //printf("\n la suma %lf",(1.0*(n_best-1-ant))/((double)COST[ant]););
            }
        }
    for (k=0;k<cl;k++){
        if (OPTIMAL_ROUTE[j+1]==NN_LIST[OPTIMAL_ROUTE[j]*cl+k]){
            PHEROMONE_MATRIX[OPTIMAL_ROUTE[j]*cl+k]+=(1.0*(n_best))/((double)BEST_GLOBAL_SOLUTION);
        }
    }
    }
}
__global__ void PHEROMONE_UPDATE_MMAS(int *ROUTE,int *BEST_ANT,
double *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,
int BEST_GLOBAL_SOLUTION){
    int j=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (j<N){
    int ant,k;
    for (k=0;k<cl;k++){
        if (OPTIMAL_ROUTE[j+1]==NN_LIST[OPTIMAL_ROUTE[j]*cl+k]){
            PHEROMONE_MATRIX[OPTIMAL_ROUTE[j]*cl+k]+=1.0/((double)BEST_GLOBAL_SOLUTION);
        }
    }
    }
}
__global__ void PHEROMONE_CHECK_MMAS(double *PHEROMONE_MATRIX,double tau_max,double tau_min){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (i<N*cl){
        if (PHEROMONE_MATRIX[i]>tau_max)PHEROMONE_MATRIX[i]=tau_max;
        if (PHEROMONE_MATRIX[i]<tau_min)PHEROMONE_MATRIX[i]=tau_min;
    }
}
__global__ void EVAPORATION(double *PHEROMONE_MATRIX,double e){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (i<N*cl){
        if (PHEROMONE_MATRIX[i]*(1-e)<FLT_MIN){
        }
        else{
             PHEROMONE_MATRIX[i]=PHEROMONE_MATRIX[i]*(1-e);
        }
    }
}
__global__ void HEURISTIC_PHEROMONE_CALCULATION(double *NODE_COORDINATE,double *PHEROMONE_MATRIX,double *HEURISTIC_PHEROMONE
,int *NN_LIST,double alpha,double beta){
    int i=blockIdx.x;
    int j=threadIdx.x; 
    if ((i<N) && (j<cl)){    
        double H=1.0/(double)EUC_2D(NODE_COORDINATE,i,NN_LIST[i*cl+j]);
        HEURISTIC_PHEROMONE[i*cl+j]=powf(H,beta)*powf(PHEROMONE_MATRIX[i*cl+j],alpha);
    }
}
__global__ void RESET_ROUTE(int *ROUTE){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);

    if (i<(N+1)*M){
        ROUTE[i]=0;
    }
}
__global__ void RESET_VISITED_LIST(bool *VISITED_LIST){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (i<N*M){
        VISITED_LIST[i]=false;
    }
}
__global__ void fijar_pheromone(double *d_pheromone,double ini_pheromone){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x); //cl
    if (i<cl*N){
        d_pheromone[i]=((double) 1)/ini_pheromone;
    }
}
__device__ int EUC_2D(double *d_d,int p1,int p2){
    double dx,dy;
    dx=(d_d[p1*2+0]-d_d[p2*2+0]);
    dy=(d_d[p1*2+1]-d_d[p2*2+1]);
    return (int)(sqrt(dx*dx+dy*dy)+0.5);                
}
__device__ void opt3(int *ROUTE,double *NODE_COORDINATE_2D,int *COST,int k,int *ROUTE_AUX,curandState *state){
    int pos[3]={};
    curandState localstate=state[k];
    for (int i=0;i<3;i++){
        int aux =0;
        while (aux==0){
            aux=1;
            float myrandf = curand_uniform(&localstate);
            myrandf *= (N-2 - 1 + 0.999999);
            myrandf += 1;
            int random = (int)truncf(myrandf);
            for (int j=0;j<3;j++){
                if (random==pos[j]){
                    aux=0;
                }
            }
            if (aux==1){
                pos[i]=random;
            }
        }
    }
    int temp;
    state[k]=localstate;
    if (pos[0]>pos[1]){
        temp=pos[1];
        pos[1]=pos[0];
        pos[0]=temp;
    } 
    if (pos[0]>pos[2]){
        temp=pos[2];
        pos[2]=pos[0];
        pos[0]=temp;
    }
    if (pos[1]>pos[2]){
        temp=pos[2];
        pos[2]=pos[1];
        pos[1]=temp;
    }  
    int delta0=pos[1]-pos[0];int delta1=pos[2]-pos[1];int aux=0;
    
    int d0,d1,d2,d3,d4,d5,d6,d7,change1,change2,change3,change4,change5,change6,change7;
    
    
    
    d0=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]-1],ROUTE[k*(N+1)+pos[0]]) +
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[1]-1],ROUTE[k*(N+1)+pos[1]]) +  
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]-1],ROUTE[k*(N+1)+pos[2]]);

    
    
    d1=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]-1],ROUTE[k*(N+1)+pos[1]-1]) + //
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]],ROUTE[k*(N+1)+pos[1]])     +  //
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]-1],ROUTE[k*(N+1)+pos[2]]);  //
    change1=d1-d0;
    
    
    d2=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]-1],ROUTE[k*(N+1)+pos[0]-1]) +//
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[1]-1],ROUTE[k*(N+1)+pos[1]])   +// 
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]],ROUTE[k*(N+1)+pos[0]]);     //
    change2=d2-d0;
    
    d3=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]-1],ROUTE[k*(N+1)+pos[0]])   +//
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[1]-1],ROUTE[k*(N+1)+pos[2]-1]) +// 
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[1]],ROUTE[k*(N+1)+pos[2]]);     //
    change3=d3-d0;
    
    d4=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]-1],ROUTE[k*(N+1)+pos[1]])   +//
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]-1],ROUTE[k*(N+1)+pos[1]-1]) +// 
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]],ROUTE[k*(N+1)+pos[2]]);     //
    change4=d4-d0;
    
    d5=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]-1],ROUTE[k*(N+1)+pos[1]-1]) +//
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]],ROUTE[k*(N+1)+pos[2]-1])   +// 
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[1]],ROUTE[k*(N+1)+pos[2]]);     //
    change5=d5-d0;
    
    d6=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]-1],ROUTE[k*(N+1)+pos[0]-1]) +//
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]],ROUTE[k*(N+1)+pos[1]-1])   +// 
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[0]],ROUTE[k*(N+1)+pos[1]]);     //
    change6=d6-d0;        
    
    d7=EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[1]],ROUTE[k*(N+1)+pos[0]-1]) +//
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[2]-1],ROUTE[k*(N+1)+pos[0]])   +// 
       EUC_2D(NODE_COORDINATE_2D,ROUTE[k*(N+1)+pos[1]-1],ROUTE[k*(N+1)+pos[2]]);     //
    change7=d7-d0;


    change7=d7-d0;                
    if (d0>d1){d0=d1;aux=1;}if (d0>d2){d0=d2;aux=2;}//if (d0>d3){d0=d3;aux=3;}
    if (d0>d4){d0=d4;aux=4;}if (d0>d5){d0=d5;aux=5;}if (d0>d6){d0=d6;aux=6;}
    if (d0>d7){d0=d7;aux=7;}
    if (aux==1){
    for (int i=0;i<pos[0];i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i]; 
    for (int i=0;i<delta0;i++)ROUTE_AUX[k*(N+1)+i+pos[0]]=ROUTE[(N+1)*k+pos[1]-i-1];
    for (int i=0;i<delta1;i++)ROUTE_AUX[k*(N+1)+i+pos[0]+delta0]=ROUTE[(N+1)*k+pos[1]+i];
    for (int i=pos[2];i<N;i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i];
    //printf(" caso 1 a' b c\n");for (int i=0;i<N+1;i++)printf("%d ",ROUTE_AUX[k*(N+1)+i]+1);
    for (int i=0;i<N;i++)ROUTE[(N+1)*k+i]=ROUTE_AUX[k*(N+1)+i];
    COST[k]+=change1;if(COST[k]<solucion)printf("\n HAY QUESO 1\n");
    
    }if (aux==2){
    for (int i=0;i<pos[0];i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i]; 
    for (int i=0;i<delta1;i++)ROUTE_AUX[k*(N+1)+i+pos[0]]=ROUTE[(N+1)*k+pos[2]-i-1];
    for (int i=0;i<delta0;i++)ROUTE_AUX[k*(N+1)+i+delta1+pos[0]]=ROUTE[(N+1)*k+pos[1]-i-1];
    for (int i=pos[2];i<N;i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i];
    //printf("\n caso 2 a b c' \n");for (int i=0;i<N+1;i++)printf("%d ",ROUTE_AUX[k*(N+1)+i]+1);
    for (int i=0;i<N;i++)ROUTE[(N+1)*k+i]=ROUTE_AUX[k*(N+1)+i];
    COST[k]+=change2;if(COST[k]<solucion)printf("\n HAY QUESO 2\n");
    }if (aux==3){
    for (int i=0;i<pos[0];i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i]; 
    for (int i=0;i<delta0;i++)ROUTE_AUX[k*(N+1)+i+pos[0]]=ROUTE[(N+1)*k+pos[0]+i];
    for (int i=0;i<delta1;i++)ROUTE_AUX[k*(N+1)+i+pos[0]+delta0]=ROUTE[(N+1)*k+pos[2]-i-1];
    for (int i=pos[2];i<N;i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i];
    //printf("\n caso 3 a b' c \n");for (int i=0;i<N+1;i++)printf("%d ",ROUTE_AUX[k*(N+1)+i]+1);
    for (int i=0;i<N;i++)ROUTE[(N+1)*k+i]=ROUTE_AUX[k*(N+1)+i];
    COST[k]+=change3;if(COST[k]<solucion)printf("\n HAY QUESO 3\n");
    }if (aux==4){
    for (int i=0;i<pos[0];i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i]; 
    for (int i=0;i<delta1;i++)ROUTE_AUX[k*(N+1)+i+pos[0]]=ROUTE[(N+1)*k+pos[1]+i];
    for (int i=0;i<delta0;i++)ROUTE_AUX[k*(N+1)+i+pos[0]+delta1]=ROUTE[(N+1)*k+pos[1]-i-1];
    for (int i=pos[2];i<N;i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i];
    //printf("\n caso 4 a' c b \n");for (int i=0;i<N+1;i++)printf("%d ",ROUTE_AUX[k*(N+1)+i]+1);
    for (int i=0;i<N;i++)ROUTE[(N+1)*k+i]=ROUTE_AUX[k*(N+1)+i];
    COST[k]+=change4;if(COST[k]<solucion)printf("\n HAY QUESO 4\n");
    }if (aux==5){
    for (int i=0;i<pos[0];i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i]; 
    for (int i=0;i<delta0;i++)ROUTE_AUX[k*(N+1)+i+pos[0]]=ROUTE[(N+1)*k+pos[1]-i-1];
    for (int i=0;i<delta1;i++)ROUTE_AUX[k*(N+1)+i+pos[0]+delta0]=ROUTE[(N+1)*k+pos[2]-i-1];
    for (int i=pos[2];i<N;i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i];
    //printf("\n caso 5 a' b' c \n");for (int i=0;i<N+1;i++)printf("%d ",ROUTE_AUX[k*(N+1)+i]+1);
    for (int i=0;i<N;i++)ROUTE[(N+1)*k+i]=ROUTE_AUX[k*(N+1)+i];
    COST[k]+=change5;if(COST[k]<solucion)printf("\n HAY QUESO 5\n");
    }if (aux==6){
    for (int i=0;i<pos[0];i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i]; 
    for (int i=0;i<delta1;i++)ROUTE_AUX[k*(N+1)+i+pos[0]]=ROUTE[(N+1)*k+pos[2]-i-1];
    for (int i=0;i<delta0;i++)ROUTE_AUX[k*(N+1)+i+pos[0]+delta1]=ROUTE[(N+1)*k+pos[0]+i];
    for (int i=pos[2];i<N;i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i];
    //printf("\n caso 6 a' b c \n");for (int i=0;i<N+1;i++)printf("%d ",ROUTE_AUX[k*(N+1)+i]+1);
    for (int i=0;i<N;i++)ROUTE[(N+1)*k+i]=ROUTE_AUX[k*(N+1)+i];
    COST[k]+=change6;if(COST[k]<solucion)printf("\n HAY QUESO 6\n");
    }if (aux==7){
    for (int i=0;i<pos[0];i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i]; 
    for (int i=0;i<delta1;i++)ROUTE_AUX[k*(N+1)+i+pos[0]]=ROUTE[(N+1)*k+pos[1]+i];
    for (int i=0;i<delta0;i++)ROUTE_AUX[k*(N+1)+i+pos[0]+delta1]=ROUTE[(N+1)*k+pos[0]+i];
    for (int i=pos[2];i<N;i++)ROUTE_AUX[k*(N+1)+i]=ROUTE[(N+1)*k+i];
    //printf("\n caso 7 a c b \n");for (int i=0;i<N+1;i++)printf("%d ",ROUTE_AUX[k*(N+1)+i]+1);
    for (int i=0;i<N;i++)ROUTE[(N+1)*k+i]=ROUTE_AUX[k*(N+1)+i];
    COST[k]+=change7;if(COST[k]<solucion)printf("\n HAY QUESO 7\n");
    }
    //printf("\n");
}
