#include "common.hpp"
#include <mpi.h>

enum class Layout { Block, Cyclic };

static int owner(int row,int n,int size,Layout l) {
    if(l==Layout::Cyclic) return row%size;
    const int q=n/size,r=n%size;
    return row < (q+1)*r ? row/(q+1) : r+(row-(q+1)*r)/q;
}

static void eliminate(std::vector<float>& a,std::vector<float>& b,int n,int rank,int size,Layout layout) {
    std::vector<float> pivot(n+1);
    for(int k=0;k<n-1;++k) {
        const int root=owner(k,n,size,layout);
        if(rank==root) { std::copy_n(&a[IDX(k,0,n)],n,pivot.data()); pivot[n]=b[k]; }
        MPI_Bcast(pivot.data(),n+1,MPI_FLOAT,root,MPI_COMM_WORLD);
        for(int i=k+1;i<n;++i) if(owner(i,n,size,layout)==rank) {
            const float f=a[IDX(i,k,n)]/pivot[k]; a[IDX(i,k,n)]=0;
            for(int j=k+1;j<n;++j) a[IDX(i,j,n)]-=f*pivot[j];
            b[i]-=f*pivot[n];
        }
    }
}

static void run(Layout layout,const char* name,const Problem& p,int rank,int size,int repeat) {
    std::vector<double> ts; std::vector<float> a,b;
    for(int r=0;r<repeat;++r) {
        a=p.a;b=p.b; MPI_Barrier(MPI_COMM_WORLD); const double t=MPI_Wtime();
        eliminate(a,b,p.n,rank,size,layout); MPI_Barrier(MPI_COMM_WORLD); ts.push_back((MPI_Wtime()-t)*1000);
    }
    for(int row=0;row<p.n;++row) MPI_Bcast(&a[IDX(row,0,p.n)],p.n,MPI_FLOAT,owner(row,p.n,size,layout),MPI_COMM_WORLD);
    for(int row=0;row<p.n;++row) MPI_Bcast(&b[row],1,MPI_FLOAT,owner(row,p.n,size,layout),MPI_COMM_WORLD);
    if(rank==0) {
        auto ra=p.a;auto rb=p.b;gauss_serial(ra,rb,p.n);
        const double e=max_abs_error(a,b,ra,rb),rr=relative_residual(p,a,b);
        print_csv(name,p.n,size,repeat,median(ts),0,0,e,rr,correct_result(e,rr));
    }
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try { const Options o=parse_options(argc,argv);const Problem p=make_problem(o.n);run(Layout::Block,"gauss_mpi_block",p,rank,size,o.repeat);run(Layout::Cyclic,"gauss_mpi_cyclic",p,rank,size,o.repeat); }
    catch(const std::exception& e){if(rank==0)std::cerr<<"gauss_mpi: "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();return 0;
}
