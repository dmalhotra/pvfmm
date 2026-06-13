/**
 * \file cheb_utils.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 3-23-2015
 * \brief This file contains implementation of Chebyshev functions.
 */

#include <cheb_utils.hpp>

namespace pvfmm{

template <>
void quad_rule<double>(int n, double* x, double* w){
  static std::vector<Vector<double> > x_lst(10000);
  static std::vector<Vector<double> > w_lst(10000);
  assert(n<10000);

  bool done=false;
  #pragma omp critical (QUAD_RULE)
  if(x_lst[n].Dim()>0){
    Vector<double>& x_=x_lst[n];
    Vector<double>& w_=w_lst[n];
    for(int i=0;i<n;i++){
      x[i]=x_[i];
      w[i]=w_[i];
    }
    done=true;
  }
  if(done) return;

  Vector<double> x_(n);
  Vector<double> w_(n);

  { //Gauss-Legendre quadrature nodes and weights, rescaled from sctl's [0,1] to [-1,1].
    sctl::Vector<double> nds, wts;
    sctl::LegQuadRule<double>::ComputeNdsWts(&nds, &wts, n);
    for(int i=0;i<n;i++){ x_[i] = 2*nds[i]-1; w_[i] = 2*wts[i]; }
  }

  #pragma omp critical (QUAD_RULE)
  { // Set x_lst, w_lst
    x_lst[n]=x_;
    w_lst[n]=w_;
  }
  quad_rule(n, x, w);
}

}//end namespace
