#include "sctl.hpp"

using namespace sctl;

template <class Real> void test_particle_fmm(const Comm& comm) {
  constexpr Integer DIM = 3;

  Stokes3D_FSxU kernel_m2l;
  Stokes3D_FxU kernel_sl;
  Stokes3D_DxU kernel_dl;
  srand48(comm.Rank());

  // Create target and source vectors.
  const Long N = 50000/comm.Size();
  Vector<Real> trg_coord(N*DIM);
  Vector<Real>  sl_coord(N*DIM);
  Vector<Real>  dl_coord(N*DIM);
  Vector<Real>  dl_norml(N*DIM);
  for (auto& a : trg_coord) a = (Real)(drand48()-0.5);
  for (auto& a :  sl_coord) a = (Real)(drand48()-0.5);
  for (auto& a :  dl_coord) a = (Real)(drand48()-0.5);
  for (auto& a :  dl_norml) a = (Real)(drand48()-0.5);
  Long n_sl  =  sl_coord.Dim()/DIM;
  Long n_dl  =  dl_coord.Dim()/DIM;

  // Set source charges.
  Vector<Real> sl_den(n_sl*kernel_sl.SrcDim());
  Vector<Real> dl_den(n_dl*kernel_dl.SrcDim());
  for (auto& a : sl_den) a = (Real)(drand48() - 0.5);
  for (auto& a : dl_den) a = (Real)(drand48() - 0.5);

  ParticleFMM<Real,DIM> fmm(comm);
  fmm.SetAccuracy(10);
  fmm.SetKernels(kernel_m2l, kernel_m2l, kernel_sl);
  fmm.AddTrg("Potential", kernel_m2l, kernel_sl);
  fmm.AddSrc("SingleLayer", kernel_sl, kernel_sl);
  fmm.AddSrc("DoubleLayer", kernel_dl, kernel_dl);
  fmm.SetKernelS2T("SingleLayer", "Potential",kernel_sl);
  fmm.SetKernelS2T("DoubleLayer", "Potential",kernel_dl);

  fmm.SetTrgCoord("Potential", trg_coord);
  fmm.SetSrcCoord("SingleLayer", sl_coord);
  fmm.SetSrcCoord("DoubleLayer", dl_coord, dl_norml);

  fmm.SetSrcDensity("SingleLayer", sl_den);
  fmm.SetSrcDensity("DoubleLayer", dl_den);

  Vector<Real> Ufmm, Uref;
  fmm.Eval(Ufmm, "Potential"); // Warm-up run
  Ufmm = 0;

  Profile::Enable(true);
  Profile::Tic("FMM-Eval", &comm);
  fmm.Eval(Ufmm, "Potential");
  Profile::Toc();
  Profile::Tic("Direct", &comm);
  fmm.EvalDirect(Uref, "Potential");
  Profile::Toc();
  Profile::print(&comm);

  Vector<Real> Uerr = Uref - Ufmm;
  { // Print error
    StaticArray<Real,2> loc_err{0,0}, glb_err{0,0};
    for (const auto& a : Uerr) loc_err[0] = std::max<Real>(loc_err[0], fabs(a));
    for (const auto& a : Uref) loc_err[1] = std::max<Real>(loc_err[1], fabs(a));
    comm.Allreduce<Real>(loc_err, glb_err, 2, CommOp::MAX);
    if (!comm.Rank()) std::cout<<"Maximum relative error: "<<glb_err[0]/glb_err[1]<<'\n';
  }
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);

  test_particle_fmm<double>(sctl::Comm::World());

  sctl::Comm::MPI_Finalize();
  return 0;
}
