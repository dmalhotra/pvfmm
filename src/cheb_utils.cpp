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

  { //Gauss-Legendre quadrature nodes and weights
    double alpha=0.0;
    double beta=0.0;
    double a=-1.0;
    double b= 1.0;
    int kind = 1;
    cgqf ( n, kind, (double)alpha, (double)beta, (double)a, (double)b, &x_[0], &w_[0] );
  }

  #pragma omp critical (QUAD_RULE)
  { // Set x_lst, w_lst
    x_lst[n]=x_;
    w_lst[n]=w_;
  }
  quad_rule(n, x, w);
}

}//end namespace
