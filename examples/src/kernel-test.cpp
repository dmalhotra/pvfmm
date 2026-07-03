// Kernel convention checks (no FMM): verifies pvfmm's Laplace potential and
// gradient kernels against analytic values, against a finite difference of
// the potential kernel, and against sctl's Laplace3D_FxU / Laplace3D_FxdU.
// In particular, gradient() must be grad(u) -- the same sign convention as
// sctl -- not -grad(u). Exits nonzero on failure.
#include <pvfmm.hpp>
#include <cstdio>
#include <cmath>

static int n_fail = 0;
static void check(const char* name, double got, double expect, double tol = 1e-8) {
  double err = fabs(got - expect) / fmax(fabs(expect), 1e-30);
  bool ok = err < tol;
  if (!ok) n_fail++;
  printf("%-34s got % .9e  expect % .9e  rel %.1e  %s\n", name, got, expect, err, ok ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  const auto& potn = pvfmm::LaplaceKernel<double>::potential();
  const auto& grad = pvfmm::LaplaceKernel<double>::gradient();

  double src[3] = {0.2, 0.3, 0.4}, qq[1] = {1.7}, trg[3] = {0.9, 0.1, 0.6};
  double rx = trg[0]-src[0], ry = trg[1]-src[1], rz = trg[2]-src[2];
  double r = sqrt(rx*rx + ry*ry + rz*rz), pi4 = 1/(4*M_PI), r3 = r*r*r;
  const double u_exact = pi4*qq[0]/r;
  const double g_exact[3] = {-pi4*qq[0]*rx/r3, -pi4*qq[0]*ry/r3, -pi4*qq[0]*rz/r3};

  { // pvfmm potential vs analytic
    double u[1] = {0};
    potn.ker_poten(src, 1, qq, 1, trg, 1, u);
    check("pvfmm potential vs analytic", u[0], u_exact, 1e-12);
  }

  double g[3] = {0,0,0};
  grad.ker_poten(src, 1, qq, 1, trg, 1, g);
  { // pvfmm gradient vs analytic grad(u)
    for (int k = 0; k < 3; k++) check("pvfmm gradient vs analytic", g[k], g_exact[k], 1e-12);
  }
  { // pvfmm gradient vs finite difference of pvfmm potential
    for (int k = 0; k < 3; k++) {
      double h = 1e-6, up[1] = {0}, um[1] = {0};
      double tp[3] = {trg[0],trg[1],trg[2]}, tm[3] = {trg[0],trg[1],trg[2]};
      tp[k] += h; tm[k] -= h;
      potn.ker_poten(src, 1, qq, 1, tp, 1, up);
      potn.ker_poten(src, 1, qq, 1, tm, 1, um);
      check("pvfmm gradient vs FD(potential)", g[k], (up[0]-um[0])/(2*h), 1e-7);
    }
  }
  { // pvfmm vs sctl
    sctl::Vector<double> Xs(3), Xt(3), F(1), Xn, U;
    for (int k = 0; k < 3; k++) { Xs[k] = src[k]; Xt[k] = trg[k]; }
    F[0] = qq[0];
    U.ReInit(1); U.SetZero();
    sctl::Laplace3D_FxU().Eval(U, Xt, Xs, Xn, F);
    double u[1] = {0};
    potn.ker_poten(src, 1, qq, 1, trg, 1, u);
    check("pvfmm potential vs sctl FxU", u[0], U[0], 1e-12);
    U.ReInit(3); U.SetZero();
    sctl::Laplace3D_FxdU().Eval(U, Xt, Xs, Xn, F);
    for (int k = 0; k < 3; k++) check("pvfmm gradient vs sctl FxdU", g[k], U[k], 1e-12);
  }

  printf(n_fail ? "FAILED (%d checks)\n" : "ALL PASS\n", n_fail);
  MPI_Finalize();
  return n_fail ? 1 : 0;
}
