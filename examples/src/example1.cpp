#include <mpi.h>
#include <omp.h>
#include <iostream>

#include <pvfmm.hpp>
#include <utils.hpp>

typedef std::vector<double> vec;

void nbody(vec&  src_coord, vec&  src_value,
           vec& surf_coord, vec& surf_value,
           vec&  trg_coord, vec&  trg_value,
					 const pvfmm::Kernel<double>& kernel_fn, MPI_Comm& comm){
  int np, rank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &rank);

  long long  n_src =  src_coord.size()/COORD_DIM;
  long long n_surf = surf_coord.size()/COORD_DIM;
  long long n_trg_glb=0, n_trg = trg_coord.size()/COORD_DIM;
  MPI_Allreduce(&n_trg , & n_trg_glb, 1, MPI_LONG_LONG, MPI_SUM, comm);

  vec glb_trg_coord(n_trg_glb*COORD_DIM);
  vec glb_trg_value(n_trg_glb*kernel_fn.ker_dim[1],0);
  std::vector<int> recv_disp(np,0);
  { // Gather all target coordinates.
    int send_cnt=n_trg*COORD_DIM;
    std::vector<int> recv_cnts(np);
    MPI_Allgather(&send_cnt    , 1, MPI_INT,
                  &recv_cnts[0], 1, MPI_INT, comm);
    pvfmm::omp_par::scan(&recv_cnts[0], &recv_disp[0], np);
    MPI_Allgatherv(&trg_coord[0]    , send_cnt                    , pvfmm::par::Mpi_datatype<double>::value(),
                   &glb_trg_coord[0], &recv_cnts[0], &recv_disp[0], pvfmm::par::Mpi_datatype<double>::value(), comm);
  }

  { // Evaluate target potential.
    vec glb_trg_value_(n_trg_glb*kernel_fn.ker_dim[1],0);
    int omp_p=omp_get_max_threads();
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t a=( i   *n_trg_glb)/omp_p;
      size_t b=((i+1)*n_trg_glb)/omp_p;

      if(kernel_fn.ker_poten!=NULL)
      kernel_fn.ker_poten(&    src_coord[0]            , n_src, &src_value[0], 1,
                          &glb_trg_coord[0]+a*COORD_DIM,   b-a, &glb_trg_value_[0]+a*kernel_fn.ker_dim[1],NULL);

      if(kernel_fn.dbl_layer_poten!=NULL)
      kernel_fn.dbl_layer_poten(&   surf_coord[0]            , n_surf, &surf_value[0], 1,
                                &glb_trg_coord[0]+a*COORD_DIM,    b-a, &glb_trg_value_[0]+a*kernel_fn.ker_dim[1],NULL);
    }
    MPI_Allreduce(&glb_trg_value_[0], &glb_trg_value[0], glb_trg_value.size(), pvfmm::par::Mpi_datatype<double>::value(), MPI_SUM, comm);
  }

  // Get local target values.
  trg_value.assign(&glb_trg_value[0]+recv_disp[rank]/COORD_DIM*kernel_fn.ker_dim[1], &glb_trg_value[0]+(recv_disp[rank]/COORD_DIM+n_trg)*kernel_fn.ker_dim[1]);
}

void fmm_test(size_t N, int mult_order, MPI_Comm comm){

  // Set kernel.
  const pvfmm::Kernel<double>& kernel_fn=pvfmm::laplace_grad_d;

  // Create target and source vectors.
  vec  trg_coord=point_distrib<double>(RandUnif,N,comm);
  vec  src_coord=point_distrib<double>(RandUnif,N,comm);
  vec surf_coord=point_distrib<double>(RandUnif,0,comm);
  size_t n_trg = trg_coord.size()/COORD_DIM;
  size_t n_src = src_coord.size()/COORD_DIM;
  size_t n_surf=surf_coord.size()/COORD_DIM;

  // Set source charges.
  vec  src_value( n_src* kernel_fn.ker_dim[0]);
  vec surf_value(n_surf*(kernel_fn.ker_dim[0]+COORD_DIM));
  for(size_t i=0;i< src_value.size();i++)  src_value[i]=drand48();
  for(size_t i=0;i<surf_value.size();i++) surf_value[i]=drand48();

  // Create memory-manager (optional)
  pvfmm::mem::MemoryManager mem_mgr(10000000);

  // Construct tree.
  size_t max_pts=300;
  pvfmm::PtFMM_Tree* tree=PtFMM_CreateTree(src_coord, src_value, surf_coord, surf_value, trg_coord, comm, max_pts, pvfmm::FreeSpace);

  // Load matrices.
  pvfmm::PtFMM matrices(&mem_mgr);
  matrices.Initialize(mult_order, comm, &kernel_fn);

  // FMM Setup
  tree->SetupFMM(&matrices);

  // Run FMM
  vec trg_value;
  PtFMM_Evaluate(tree, trg_value, n_trg);

  // Re-run FMM
  tree->ClearFMMData();
  for(size_t i=0;i< src_value.size();i++)  src_value[i]=drand48();
  for(size_t i=0;i<surf_value.size();i++) surf_value[i]=drand48();
  PtFMM_Evaluate(tree, trg_value, n_trg, &src_value, &surf_value);

  {// Check error
    vec trg_sample_coord;
    vec trg_sample_value;
    size_t n_trg_sample=0;
    { // Sample target points for verifications.
      size_t n_skip=N*n_trg/1e9;
      if(!n_skip) n_skip=1;
      for(size_t i=0;i<n_trg;i=i+n_skip){
        for(size_t j=0;j<COORD_DIM;j++)
          trg_sample_coord.push_back(trg_coord[i*COORD_DIM+j]);
        for(size_t j=0;j<kernel_fn.ker_dim[1];j++)
          trg_sample_value.push_back(trg_value[i*kernel_fn.ker_dim[1]+j]);
        n_trg_sample++;
      }
    }

    // Direct n-body
    vec trg_sample_value_(n_trg_sample*kernel_fn.ker_dim[1]);
    nbody(       src_coord,        src_value ,
                surf_coord,       surf_value ,
          trg_sample_coord, trg_sample_value_, kernel_fn, comm);

    // Compute error
    double max_err=0, max_val=0;
    double max_err_glb=0, max_val_glb=0;
    for(size_t i=0;i<n_trg_sample*kernel_fn.ker_dim[1];i++){
      if(fabs(trg_sample_value_[i]-trg_sample_value[i])>max_err)
        max_err=fabs(trg_sample_value_[i]-trg_sample_value[i]);
      if(fabs(trg_sample_value_[i])>max_val)
        max_val=fabs(trg_sample_value_[i]);
    }
    MPI_Reduce(&max_err, &max_err_glb, 1, pvfmm::par::Mpi_datatype<double>::value(), MPI_MAX, 0, comm);
    MPI_Reduce(&max_val, &max_val_glb, 1, pvfmm::par::Mpi_datatype<double>::value(), MPI_MAX, 0, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if(!rank) std::cout<<"Maximum Absolute Error:"<<max_err_glb<<'\n';
    if(!rank) std::cout<<"Maximum Relative Error:"<<max_err_glb/max_val_glb<<'\n';
  }

  // Free memory
  delete tree;
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  MPI_Comm comm=MPI_COMM_WORLD;

  // Read command line options.
  commandline_option_start(argc, argv, "\
  This example demonstrates solving a particle N-body problem,\n\
with Laplace Gradient kernel, using the PvFMM library.\n");
  commandline_option_start(argc, argv);
  omp_set_num_threads( atoi(commandline_option(argc, argv,  "-omp",     "1", false, "-omp  <int> = (1)    : Number of OpenMP threads."          )));
  size_t   N=(size_t)strtod(commandline_option(argc, argv,    "-N",     "1",  true, "-N    <int>          : Number of source and target points."),NULL);
  int      m=       strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  commandline_option_end(argc, argv);
  pvfmm::Profile::Enable(true);

  // Run FMM with above options.
  fmm_test(N, m, comm);

  //Output Profiling results.
  pvfmm::Profile::print(&comm);

  // Shut down MPI
  MPI_Finalize();
  return 0;
}

