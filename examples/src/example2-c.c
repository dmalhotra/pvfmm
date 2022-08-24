#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pvfmm.h>

void fn_input(const double* coord, long n, double* out, void* ctx){ //Input function
  int dof=3;
  double L=125;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*3];
    {
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=                                        0+2*L*exp(-L*r_2)*(c[0]-0.5);
      out[i*dof+1]= 4*L*L*(c[2]-0.5)*(5-2*L*r_2)*exp(-L*r_2)+2*L*exp(-L*r_2)*(c[1]-0.5);
      out[i*dof+2]=-4*L*L*(c[1]-0.5)*(5-2*L*r_2)*exp(-L*r_2)+2*L*exp(-L*r_2)*(c[2]-0.5);
    }
  }
}
void fn_poten(const double* coord, long n, double* out, void* ctx){ //Output potential
  int dof=3;
  double L=125;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*3];
    {
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]= 0;
      out[i*dof+1]= 2*L*(c[2]-0.5)*exp(-L*r_2);
      out[i*dof+2]=-2*L*(c[1]-0.5)*exp(-L*r_2);
    }
  }
}

void GetChebNodes(double* cheb_coord, long Nleaf, int cheb_deg, int depth, const double* leaf_coord) {
  const double leaf_length = 1./(1<<depth);
  const int Ncheb = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
  for (long leaf_idx = 0; leaf_idx < Nleaf; leaf_idx++) {
    for (long j2 = 0; j2 < cheb_deg+1; j2++) {
      for (long j1 = 0; j1 < cheb_deg+1; j1++) {
        for (long j0 = 0; j0 < cheb_deg+1; j0++) {
          const long node_idx = leaf_idx * Ncheb + (j2 * (cheb_deg+1) + j1) * (cheb_deg+1) + j0;
          cheb_coord[node_idx*3+0] = leaf_coord[leaf_idx*3+0] + (1-cos(M_PI*(j0*2+1)/(cheb_deg*2+2))) * leaf_length * 0.5;
          cheb_coord[node_idx*3+1] = leaf_coord[leaf_idx*3+1] + (1-cos(M_PI*(j1*2+1)/(cheb_deg*2+2))) * leaf_length * 0.5;
          cheb_coord[node_idx*3+2] = leaf_coord[leaf_idx*3+2] + (1-cos(M_PI*(j2*2+1)/(cheb_deg*2+2))) * leaf_length * 0.5;
        }
      }
    }
  }
}

void test1(void* fmm, int kdim0, int kdim1, int cheb_deg, const MPI_Comm comm) { // Build volume tree using function pointer
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  srand48(mpi_rank);

  const long Nt = 100;
  double* trg_coord = (double*)malloc(Nt*3*sizeof(double));
  double* trg_value = (double*)malloc(Nt*kdim1*sizeof(double));
  double* trg_value_ref = (double*)malloc(Nt*kdim1*sizeof(double));
  for (long i = 0; i < Nt*3; i++) trg_coord[i] = drand48();
  fn_poten(trg_coord, Nt, trg_value_ref, NULL);

  // Build volume tree
  void* tree = PVFMMCreateVolumeTreeD(cheb_deg, kdim0, fn_input, NULL, trg_coord, Nt, comm, 1e-6, 100, false, 0);

  // Evaluate FMM
  PVFMMEvaluateVolumeFMMD(trg_value, tree, fmm, Nt);

  { // Print error
    double max_err = 0, max_val = 0;
    double max_err_glb = 0, max_val_glb = 0;
    for (long i = 0; i < Nt*kdim1; i++) {
      max_err = fmax(max_err, fabs(trg_value[i]-trg_value_ref[i]));
      max_val = fmax(max_val, fabs(trg_value_ref[i]));
    }
    MPI_Reduce(&max_err, &max_err_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&max_val, &max_val_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (!mpi_rank) printf("Maximum relative error = %e\n", max_err_glb/max_val_glb);
  }

  PVFMMDestroyVolumeTreeD(&tree);
  free(trg_coord);
  free(trg_value);
  free(trg_value_ref);
}

void test2(void* fmm, int kdim0, int kdim1, int cheb_deg, const MPI_Comm comm) { // Build volume tree from Chebyshev coefficients
  const int Ncheb = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
  const int Ncoef = (cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
  srand48(mpi_rank);

  void* tree;
  const int depth = 3;
  { // Build uniform tree
    const long Nleaf = (1<<(3*depth));
    const long Nleaf_loc = Nleaf*(mpi_rank+1)/mpi_size - Nleaf*mpi_rank/mpi_size;
    const double leaf_length = 1./(1<<depth);

    double* leaf_coord = (double*)malloc(Nleaf_loc*3*sizeof(double));
    for (long leaf_idx = 0; leaf_idx < Nleaf_loc; leaf_idx++) {
      const long leaf_idx_glb = Nleaf*mpi_rank/mpi_size + leaf_idx;
      leaf_coord[leaf_idx*3+0] = (leaf_idx_glb/(1<<depth*0))%(1<<depth) * leaf_length;
      leaf_coord[leaf_idx*3+1] = (leaf_idx_glb/(1<<depth*1))%(1<<depth) * leaf_length;
      leaf_coord[leaf_idx*3+2] = (leaf_idx_glb/(1<<depth*2))%(1<<depth) * leaf_length;
    }

    double* cheb_coord = (double*)malloc(Nleaf_loc*Ncheb*3*sizeof(double));
    double* dens_value = (double*)malloc(Nleaf_loc*Ncheb*kdim0*sizeof(double));
    double* dens_coeff = (double*)malloc(Nleaf_loc*Ncoef*kdim0*sizeof(double));
    GetChebNodes(cheb_coord, Nleaf_loc, cheb_deg, depth, leaf_coord);
    fn_input(cheb_coord, Nleaf_loc*Ncheb, dens_value, NULL);
    PVFMMNodes2CoeffD(dens_coeff, Nleaf_loc, cheb_deg, kdim0, dens_value);
    tree = PVFMMCreateVolumeTreeFromCoeffD(Nleaf_loc, cheb_deg, kdim0, leaf_coord, dens_coeff, NULL, 0, comm, false);

    free(leaf_coord);
    free(cheb_coord);
    free(dens_value);
    free(dens_coeff);
  }

  // Evaluate FMM
  PVFMMEvaluateVolumeFMMD(NULL, tree, fmm, 0);

  // Get potential at Chebyshev nodes
  const long Nleaf = PVFMMGetLeafCountD(tree);
  double* potn_coeff = (double*)malloc(Nleaf*Ncoef*kdim1*sizeof(double));
  double* potn_value = (double*)malloc(Nleaf*Ncheb*kdim1*sizeof(double));
  PVFMMGetPotentialCoeffD(potn_coeff, tree);
  PVFMMCoeff2NodesD(potn_value, Nleaf, cheb_deg, kdim1, potn_coeff);

  // Get reference solution at Chebyshev nodes
  double* leaf_coord = (double*)malloc(Nleaf*3*sizeof(double));
  double* cheb_coord = (double*)malloc(Nleaf*Ncheb*3*sizeof(double));
  double* potn_value_ref = (double*)malloc(Nleaf*Ncheb*kdim1*sizeof(double));
  PVFMMGetLeafCoordD(leaf_coord, tree);
  GetChebNodes(cheb_coord, Nleaf, cheb_deg, depth, leaf_coord);
  fn_poten(cheb_coord, Nleaf*Ncheb, potn_value_ref, NULL);

  { // Print error
    double max_err = 0, max_val = 0;
    double max_err_glb = 0, max_val_glb = 0;
    for (long i = 0; i < Nleaf*Ncheb*kdim1; i++) {
      max_err = fmax(max_err, fabs(potn_value[i]-potn_value_ref[i]));
      max_val = fmax(max_val, fabs(potn_value_ref[i]));
    }
    MPI_Reduce(&max_err, &max_err_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&max_val, &max_val_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (!mpi_rank) printf("Maximum relative error = %e\n", max_err_glb/max_val_glb);
  }

  PVFMMDestroyVolumeTreeD(&tree);
  free(potn_coeff);
  free(potn_value);
  free(leaf_coord);
  free(cheb_coord);
  free(potn_value_ref);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  const MPI_Comm comm = MPI_COMM_WORLD;
  const int mult_order = 10, cheb_deg = 14, kdim0 = 3, kdim1 = 3;

  // Build FMM translation operators
  void* fmm = PVFMMCreateVolumeFMMD(mult_order, cheb_deg, PVFMMStokesVelocity, comm);

  //test1(fmm, kdim0, kdim1, cheb_deg, comm);
  test2(fmm, kdim0, kdim1, cheb_deg, comm);

  PVFMMDestroyVolumeFMMD(&fmm);

  MPI_Finalize();
  return 0;
}

