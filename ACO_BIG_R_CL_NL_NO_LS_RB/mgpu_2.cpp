#include "mgpu_2.h"
void lectura_2(double *dis){
    FILE* f;
    int height, width, ii, jj;
    char buff[100]; 
    if((f = fopen(name_e, "r")) == NULL)
        exit(1);    
    for (ii=0;ii<6;ii++){
        fscanf(f,"%[^\n]\n",&buff);
        printf("%s\n" ,buff);
    } 
    for(ii=0; ii<N; ii++)
        if(fscanf(f, "%*i %lf %lf",&dis[ii*2+0],&dis[ii*2+1]) != 2)
            exit(1);
    fclose(f);
    //for (ii=0;ii<N;ii++)printf("%i %i %i\n",dis[ii*3+0],dis[ii*3+1],dis[ii*3+2]);
}
void guardar_iteration_time(double *time,double alpha,double beta,double e){
    FILE *file1;
    file1 = fopen(name_test_1, "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
        fprintf(file1,"%d,%d,%lf,%lf,%lf,",N,4*M,alpha,beta,e);
        for (int i=0;i<N_e;i++){
            if (i==0){
                fprintf(file1,"%lf",time[i]);
            }
            else{
                fprintf(file1,",%lf",time[i]);
            }
        }
        fprintf(file1,"\n");
    }
    fclose(file1);
}
void guardar_iteration_time_series(double *time,double alpha,double beta,double e){
    FILE *file1; 
    file1 = fopen(name_test_2, "a");
    int i;
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else {
        fprintf(file1,"%d,%d,%lf,%lf,%lf,",N,4*M,alpha,beta,e);
        for (i=1;i<ITERACION;i++){
            if (i==0){
                fprintf(file1,"%lf",time[i]);
            }
            else{
                fprintf(file1,",%lf",time[i]);
            }
        }
        fprintf(file1,"\n"); 
    }

}
void guardar_warm_up(double *time,double alpha,double beta,double e){
    FILE *file1;
    file1 = fopen(name_test_3, "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
        fprintf(file1,"%d,%d,%lf,%lf,%lf,",N,4*M,alpha,beta,e);
        for (int i=0;i<N_e;i++){
            if (i==0){
                fprintf(file1,"%lf",time[i]);
            }
            else{
                fprintf(file1,",%lf",time[i]);
            }
        }
        fprintf(file1,"\n");
    }
    fclose(file1);
}
void guardar_soluciones(int *soluciones,double alpha,double beta,double e){
    FILE *file1;
    file1 = fopen(name_test_4, "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
        fprintf(file1,"%d,%d,%lf,%lf,%lf,",N,4*M,alpha,beta,e);
        for (int i=0;i<N_e;i++){
            if (i==0){
                fprintf(file1,"%d",soluciones[i]);
            }
            else{
                fprintf(file1,",%d",soluciones[i]);
            }
        }
        fprintf(file1,"\n");
    }
    fclose(file1);
}

double promediar_int(int *vec){
    double sumaprom=0;
    for (int i=0;i<N_e;i++){
        sumaprom+=(double)vec[i];
    }
    return sumaprom/N_e;
}
double std_vec(double *vec, double prom){
    double suma_std=0;
    for (int i=0;i<N_e;i++){
        suma_std+=powf(vec[i]-prom,2);
    }
    suma_std=suma_std/N_e;
    suma_std=powf(suma_std,0.5);
    return suma_std;
}
double std_vec_it(int *vec, double prom){
    double suma_std=0;
    for (int i=0;i<N_e;i++){
        suma_std+=powf((double)vec[i]-prom,2);
    }
    suma_std=suma_std/N_e;
    suma_std=powf(suma_std,0.5);
    return suma_std;
}
int EUC_2D_C(double *d_d,int p1,int p2){
    double dx,dy;
    dx=(d_d[p1*2+0]-d_d[p2*2+0]);
    dy=(d_d[p1*2+1]-d_d[p2*2+1]);
    return (int)(sqrt(dx*dx+dy*dy)+0.5);                
}
int rutainicial(int *rute_op,double *d,bool *lista_vis){
    rute_op[0]=0;
    int cost=0;
    for (int i=0;i<N-1;i++){
        lista_vis[rute_op[i]]=true;
        int min=100000000;
        int mink=0;
        if (i%1000==0)printf("\n voy en el nodo %d \n",i);
        for (int j=0;j<N;j++){
            if (rute_op[i]!=j){
                if (lista_vis[j]==false){
                    //d[rute_op[i]*N+j]
                if (EUC_2D_C(d,rute_op[i],j)<min){
                    min=EUC_2D_C(d,rute_op[i],j);
                    mink=j;
                }
            }
            }
        }
        rute_op[i+1]=mink;
        cost+=EUC_2D_C(d,rute_op[i],mink);
    }
    cost+=EUC_2D_C(d,rute_op[N-1],0);
    printf("\n");
    rute_op[N]=0;
    //for (int i=0;i<N+1;i++)printf("%d ",rute_op[i]);
    printf("\n el coste incial es %d \n",cost);
    return cost;
}

int mejorhormiga(double *cost,int *best_horm,double *best_cost){
    double min=cost[0];
    int horm_op;
    for (int i=0;i<M;i++){
        best_horm[i]=i;
        best_cost[i]=cost[i];
        if (cost[i]<=min){
            min=cost[i];
            horm_op=i;
        }
    }
    double temp;
    int temp1;
    for(int i = 0; i<M; i++) {
        for(int j = i+1; j<M; j++)
        {
            if(best_cost[j] < best_cost[i]) {
                temp1=best_horm[i];
            best_horm[i]=best_horm[j];
            best_horm[j]=temp1; 
            temp = best_cost[i];
                best_cost[i] = best_cost[j];
                best_cost[j] = temp;
            }
        }
    }
    return horm_op;
}
double minimovec(double *vec){
    double min=vec[0];
    for (int i=0;i<N_e;i++){
        if (min>=vec[i]){
            min=vec[i];
        }
    }
    return min;
}
double promediar(double *vec){
    double sumaprom=0;
    for (int i=0;i<N_e;i++){
        sumaprom+=vec[i];
    }
    return sumaprom/N_e;
}
double opt33(int *rute_op,int *rute_op_aux,double *d,double global_sol){
    
    int pos[3]={};
    for (int i=0;i<3;i++){
        int aux =0;
        while (aux==0){
            aux=1;
            int random = (int) rand()%(N-2)+1;
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
    //printf("\n array \n %d %d %d",pos[0],pos[1],pos[2]);
    int delta0=pos[1]-pos[0];int delta1=pos[2]-pos[1];int aux=0;
    
    double d0,d1,d2,d3,d4,d5,d6,d7,change1,change2,change3,change4,change5,change6,change7;
    d0=d[rute_op[pos[0]-1]*N+rute_op[pos[0]]]   +      d[rute_op[pos[1]-1]*N+rute_op[pos[1]]]   +    d[rute_op[pos[2]-1]*N+rute_op[pos[2]]];
    d1=d[rute_op[pos[0]-1]*N+rute_op[pos[1]-1]]   +      d[rute_op[pos[0]]*N+rute_op[pos[1]]]   +    d[rute_op[pos[2]-1]*N+rute_op[pos[2]]];change1=d1-d0;
    d2=d[rute_op[pos[2]-1]*N+rute_op[pos[0]-1]]   +      d[rute_op[pos[1]-1]*N+rute_op[pos[1]]]   +    d[rute_op[pos[2]]*N+rute_op[pos[0]]];change2=d2-d0;
    d3=d[rute_op[pos[0]-1]*N+rute_op[pos[0]]]   +      d[rute_op[pos[1]-1]*N+rute_op[pos[2]-1]]   +    d[rute_op[pos[1]]*N+rute_op[pos[2]]];change3=d3-d0;
    d4=d[rute_op[pos[0]-1]*N+rute_op[pos[1]]]   +      d[rute_op[pos[2]-1]*N+rute_op[pos[1]-1]]   +    d[rute_op[pos[0]]*N+rute_op[pos[2]]];change4=d4-d0;
    d5=d[rute_op[pos[0]-1]*N+rute_op[pos[1]-1]]   +      d[rute_op[pos[0]]*N+rute_op[pos[2]-1]]   +    d[rute_op[pos[1]]*N+rute_op[pos[2]]];change5=d5-d0;
    d6=d[rute_op[pos[2]-1]*N+rute_op[pos[0]-1]]   +      d[rute_op[pos[2]]*N+rute_op[pos[1]-1]]   +    d[rute_op[pos[0]]*N+rute_op[pos[1]]];change6=d6-d0;
    d7=d[rute_op[pos[1]]*N+rute_op[pos[0]-1]]   +      d[rute_op[pos[2]-1]*N+rute_op[pos[0]]]   +    d[rute_op[pos[1]-1]*N+rute_op[pos[2]]];change7=d7-d0;
    if (d0>d1){d0=d1;aux=1;}if (d0>d2){d0=d2;aux=2;}if (d0>d3){d0=d3;aux=3;}
    if (d0>d4){d0=d4;aux=4;}if (d0>d5){d0=d5;aux=5;}if (d0>d6){d0=d6;aux=6;}
    if (d0>d7){d0=d7;aux=7;}
    if (aux==1){
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]-i-1];
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]+delta0]=rute_op[pos[1]+i];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf(" caso 1 a' b c\n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%lf \n",global_sol+change1);
    return global_sol+change1;
    }if (aux==2){
    for (int i=0;i<N;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[2]-i-1];
    for (int i=0;i<delta0;i++)rute_op_aux[i+delta1+pos[0]]=rute_op[pos[1]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 2 a b c' \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%lf \n",global_sol+change2);
    return global_sol+change2;
    }if (aux==3){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]]=rute_op[pos[0]+i];
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]+delta0]=rute_op[pos[2]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 3 a b' c \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%lf \n",global_sol+change3);
    return global_sol+change3;
    }if (aux==4){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]+i];
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]+delta1]=rute_op[pos[1]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 4 a' c b \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%lf \n",global_sol+change4);
    return global_sol+change4;
    }if (aux==5){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]-i-1];
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]+delta0]=rute_op[pos[2]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 5 a' b' c \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%lf \n",global_sol+change5);
    return global_sol+change5;
    }if (aux==6){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[2]-i-1];
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]+delta1]=rute_op[pos[0]+i];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 6 a' b c \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%lf \n",global_sol+change6);
    return global_sol+change6;
    }if (aux==7){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]+i];
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]+delta1]=rute_op[pos[0]+i];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 7 a c b \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%lf \n",global_sol+change7);
    return global_sol+change7;
    }
    if (aux==0){
        //printf("no ha pasao na\n");
        //printf("d0 %lf d1 %lf d2 %lf d3 %lf d4 %lf d5 %lf d6 %lf d7 %lf",d0,d1,d2,d3,d4,d5,d6,d7);
        return global_sol;
    }
    //printf("\n");
    return global_sol;
}
