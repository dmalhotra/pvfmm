#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pvfmm.h>

void BiotSavart(const double* src_X, const double* src_V, long Ns, const double* trg_X, double* trg_V, long Nt) { // Direct evaluation
  double oofp = 1 / (4 * M_PI);
  #pragma omp parallel for schedule(static)
  for (long t = 0; t < Nt; t++) {
    trg_V[t*3+0] = 0;
    trg_V[t*3+1] = 0;
    trg_V[t*3+2] = 0;
    for (long s = 0; s < Ns; s++) {
      double X[3];
      X[0] = trg_X[t*3+0] - src_X[s*3+0];
      X[1] = trg_X[t*3+1] - src_X[s*3+1];
      X[2] = trg_X[t*3+2] - src_X[s*3+2];
      double r2 = X[0]*X[0] + X[1]*X[1] + X[2]*X[2];
      double rinv = (r2 > 0 ? 1/sqrt(r2) : 0);
      double rinv3 = rinv * rinv * rinv;

      trg_V[t*3+0] += (src_V[s*3+1]*X[2] - src_V[s*3+2]*X[1]) * rinv3 * oofp;
      trg_V[t*3+1] += (src_V[s*3+2]*X[0] - src_V[s*3+0]*X[2]) * rinv3 * oofp;
      trg_V[t*3+2] += (src_V[s*3+0]*X[1] - src_V[s*3+1]*X[0]) * rinv3 * oofp;
    }
  }
}

void test_FMM(void* ctx) { // Compare FMM and direct evaluation results
  int kdim[2] = {3,3};
  long Ns = 20000;
  long Nt = 20000;

  // Initialize target coordinates
  double* trg_X  = (double*)malloc(Nt *       3 * sizeof(double));
  double* trg_V  = (double*)malloc(Nt * kdim[1] * sizeof(double));
  double* trg_V0 = (double*)malloc(Nt * kdim[1] * sizeof(double));
  for (long i = 0; i < Nt * 3; i++) trg_X[i] = drand48();

  // Initialize source coordinates and density
  double* src_X  = (double*)malloc(Ns *       3 * sizeof(double));
  double* src_V  = (double*)malloc(Ns * kdim[0] * sizeof(double));
  for (long i = 0; i < Ns *       3; i++) src_X[i] = drand48();
  for (long i = 0; i < Ns * kdim[0]; i++) src_V[i] = drand48() - 0.5;

  // FMM evaluation
  double tt;
  int setup = 1;
  tt = -omp_get_wtime();
  PVFMMEvalD(src_X, src_V, NULL, Ns, trg_X, trg_V, Nt, ctx, setup);
  printf("FMM evaluation time (with setup) : %f\n", tt + omp_get_wtime());

  // FMM evaluation (without setup)
  setup = 0;
  tt = -omp_get_wtime();
  PVFMMEvalD(src_X, src_V, NULL, Ns, trg_X, trg_V, Nt, ctx, setup);
  printf("FMM evaluation time (without setup) : %f\n", tt + omp_get_wtime());

  // Direct evaluation
  tt = -omp_get_wtime();
  BiotSavart(&src_X[0], &src_V[0], Ns, &trg_X[0], &trg_V0[0], Nt);
  printf("Direct evaluation time : %f\n", tt + omp_get_wtime());

  double max_err = 0, max_val = 0;
  for (long i = 0; i < Nt * kdim[1]; i++) { // Compute error
    double val = fabs(trg_V0[i]), err = fabs(trg_V[i] - trg_V0[i]);
    if (val > max_val) max_val = val;
    if (err > max_err) max_err = err;
  }
  printf("Max relative error : %e\n", max_err / max_val);

  free(src_X);
  free(src_V);
  free(trg_X);
  free(trg_V);
  free(trg_V0);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  void* ctx;
  { // Create FMM context
    double box_size = -1;
    int points_per_box = 1000;
    int multipole_order = 10;
    enum PVFMMKernel kernel = PVFMMBiotSavartPotential;
    ctx = PVFMMCreateContextD(box_size, points_per_box, multipole_order, kernel, MPI_COMM_WORLD);
  }

  test_FMM(ctx);

  PVFMMDestroyContextD(&ctx);

  MPI_Finalize();
  return 0;
}

