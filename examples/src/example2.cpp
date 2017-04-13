#include <iostream>
#include <mpi.h>
#include <omp.h>

#include <pvfmm.hpp>
#include <utils.hpp>

PeriodicType periodicType = PeriodicType::NONE;

// Input function
void fn_input(const double *coord, int n, double *out) {
  double a = -160;
  for (int i = 0; i < n; i++) {
    const double *c = &coord[i * 3];
    double r_2 = (c[0] - 0.5) * (c[0] - 0.5) + (c[1] - 0.5) * (c[1] - 0.5) +
                 (c[2] - 0.5) * (c[2] - 0.5);
    out[i] = (2 * a * r_2 + 3) * 2 * a * exp(a * r_2);
  }
}

// Analytical solution (Expected output)
void fn_output(const double *coord, int n, double *out) {
  double a = -160;
  for (int i = 0; i < n; i++) {
    const double *c = &coord[i * 3];
    double r_2 = (c[0] - 0.5) * (c[0] - 0.5) + (c[1] - 0.5) * (c[1] - 0.5) +
                 (c[2] - 0.5) * (c[2] - 0.5);
    out[i] = -exp(a * r_2);
  }
}

void fmm_test(size_t N, int mult_order, int cheb_deg, double tol,
              MPI_Comm comm) {

  // Set kernel.
  const pvfmm::Kernel<double> &kernel_fn =
      pvfmm::LaplaceKernel<double>::potential();

  // Construct tree.
  size_t max_pts = 100;
  std::vector<double> trg_coord = point_distrib<double>(RandUnif, N, comm);
  pvfmm::ChebFMM_Tree *tree =
      ChebFMM_CreateTree(cheb_deg, kernel_fn.ker_dim[0], fn_input, trg_coord,
                         comm, tol, max_pts, pvfmm::FreeSpace);

  // Load matrices.
  pvfmm::ChebFMM matrices;
  matrices.Initialize(mult_order, cheb_deg, comm, &kernel_fn);

  // FMM Setup
  tree->SetupFMM(&matrices);

  // Run FMM
  std::vector<double> trg_value;
  size_t n_trg = trg_coord.size() / COORD_DIM;
  pvfmm::ChebFMM_Evaluate(tree, trg_value, n_trg);

  // Re-run FMM
  tree->ClearFMMData();
  pvfmm::ChebFMM_Evaluate(tree, trg_value, n_trg);

  { // Check error
    std::vector<double> trg_value_(n_trg * kernel_fn.ker_dim[1]);
    fn_output(&trg_coord[0], n_trg, &trg_value_[0]);
    double max_err = 0;
    double max_err_glb = 0;
    for (size_t i = 0; i < n_trg; i++) {
      if (fabs(trg_value_[i] - trg_value[i]) > max_err)
        max_err = fabs(trg_value_[i] - trg_value[i]);
    }
    MPI_Reduce(&max_err, &max_err_glb, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (!rank)
      std::cout << "Maximum Error:" << max_err_glb << '\n';
  }

  // Free memory
  delete tree;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  // Read command line options.
  commandline_option_start(argc, argv, "\
  This example demonstrates solving a volume potential problem,\n\
with Laplace kernel, using the PvFMM library.\n");
  commandline_option_start(argc, argv);
  omp_set_num_threads(atoi(
      commandline_option(argc, argv, "-omp", "1", false,
                         "-omp  <int> = (1)    : Number of OpenMP threads.")));
  size_t N = (size_t)strtod(
      commandline_option(argc, argv, "-N", "1", true,
                         "-N    <int>          : Number of target points."),
      NULL);
  int m =
      strtoul(commandline_option(
                  argc, argv, "-m", "10", false,
                  "-m    <int> = (10)   : Multipole order (+ve even integer)."),
              NULL, 10);
  int q = strtoul(commandline_option(
                      argc, argv, "-q", "14", false,
                      "-q    <int> = (14)   : Chebyshev order (+ve integer)."),
                  NULL, 10);
  double tol =
      strtod(commandline_option(
                 argc, argv, "-tol", "1e-5", false,
                 "-tol <real> = (1e-5) : Tolerance for adaptive refinement."),
             NULL);
  commandline_option_end(argc, argv);
  pvfmm::Profile::Enable(true);

  // Run FMM with above options.
  fmm_test(N, m, q, tol, comm);

  // Output Profiling results.
  pvfmm::Profile::print(&comm);

  // Shut down MPI
  MPI_Finalize();
  return 0;
}
