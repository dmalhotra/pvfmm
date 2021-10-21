#include <mpi.h>
#include <omp.h>
#include <iostream>

#include <pvfmm.hpp>
#include <utils.hpp>

typedef std::vector<double> vec;

void fmm_test(size_t N, int mult_order, MPI_Comm comm){
  // Set kernel.
  const pvfmm::Kernel<double>& kernel_fn_old = pvfmm::StokesKernel<double>::vel_grad();
  const pvfmm::Kernel<double>& kernel_fn_new = pvfmm::StokesKernelNew<double>::vel_grad();

  // Create target and source vectors.
  vec trg_coord = point_distrib<double>(RandUnif, N, comm);
  vec sl_coord = point_distrib<double>(RandUnif, N, comm);
  vec dl_coord = point_distrib<double>(RandUnif, 0, comm);
  size_t n_trg = trg_coord.size() / PVFMM_COORD_DIM;
  size_t n_sl = sl_coord.size() / PVFMM_COORD_DIM;
  size_t n_dl = dl_coord.size() / PVFMM_COORD_DIM;

  // Set source charges.
  vec sl_den(n_sl * kernel_fn_old.ker_dim[0]);
  vec dl_den(n_dl * (kernel_fn_old.ker_dim[0] + PVFMM_COORD_DIM));
  for (size_t i = 0; i < sl_den.size(); i++) sl_den[i] = drand48();
  for (size_t i = 0; i < dl_den.size(); i++) dl_den[i] = drand48();

  // Create memory-manager (optional)
  pvfmm::mem::MemoryManager mem_mgr(10000000);

  // Construct tree.
  size_t max_pts = 600;
  auto* tree_old = PtFMM_CreateTree(sl_coord, sl_den, dl_coord, dl_den, trg_coord, comm, max_pts, pvfmm::FreeSpace);
  auto* tree_new = PtFMM_CreateTree(sl_coord, sl_den, dl_coord, dl_den, trg_coord, comm, max_pts, pvfmm::FreeSpace);

  // Load matrices.
  pvfmm::PtFMM<double> matrices_old(&mem_mgr);
  matrices_old.Initialize(mult_order, comm, &kernel_fn_old);
  pvfmm::PtFMM<double> matrices_new(&mem_mgr);
  matrices_new.Initialize(mult_order, comm, &kernel_fn_new);

  // FMM Setup
  tree_old->SetupFMM(&matrices_old);
  tree_new->SetupFMM(&matrices_new);

  // Run FMM
  vec trg_value_old, trg_value_new;
  PtFMM_Evaluate(tree_old, trg_value_old, n_trg);
  PtFMM_Evaluate(tree_new, trg_value_new, n_trg);

  {// Check error
    // Compute error
    double max_err = 0, max_rel_err = 0;
    for (size_t i = 0; i < trg_value_old.size(); i++) {
      if (fabs(trg_value_new[i] - trg_value_old[i]) > max_err)
          max_err = fabs(trg_value_new[i] - trg_value_old[i]);
      if (trg_value_old[i] != 0.0 && fabs(trg_value_new[i] / trg_value_old[i] - 1.0) > max_rel_err)
          max_rel_err = fabs(trg_value_new[i] / trg_value_old[i] - 1.0);
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &max_err, &max_err, 1, pvfmm::par::Mpi_datatype<double>::value(), MPI_MAX, 0, comm);
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &max_rel_err, &max_rel_err, 1, pvfmm::par::Mpi_datatype<double>::value(), MPI_MAX, 0, comm);

    if (!rank) std::cout << "Maximum Absolute Error:" << max_err << '\n';
    if (!rank) std::cout << "Maximum Relative Error:" << max_rel_err << '\n';
  }

  // Free memory
  delete tree_old;
  delete tree_new;
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  // Read command line options.
  commandline_option_start(argc, argv, "\
  This example demonstrates solving a particle N-body problem,\n\
with Laplace Gradient kernel, using the PvFMM library.\n");
  omp_set_num_threads( atoi(commandline_option(argc, argv,  "-omp",     "1", false, "-omp  <int> = (1)    : Number of OpenMP threads."          )));
  size_t   N=(size_t)strtod(commandline_option(argc, argv,    "-N",     "1",  true, "-N    <int>          : Number of source and target points."),NULL);
  int      m=       strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  commandline_option_end(argc, argv);
  // pvfmm::Profile::Enable(true);

  // Run FMM with above options.
  fmm_test(N, m, comm);

  //Output Profiling results.
  // pvfmm::Profile::print(&comm);

  // Shut down MPI
  MPI_Finalize();
  return 0;
}

