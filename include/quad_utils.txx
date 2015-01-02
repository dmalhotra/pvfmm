/**
 * \file quad_utils.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 7-16-2014
 * \brief This file contains quadruple-precision related functions.
 */

#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

QuadReal_t atoquad(const char* str){
  int i=0;
  QuadReal_t sign=1.0;
  for(;str[i]!='\0';i++){
    char c=str[i];
    if(c=='-') sign=-sign;
    if(c>='0' && c<='9') break;
  }

  QuadReal_t val=0.0;
  for(;str[i]!='\0';i++){
    char c=str[i];
    if(c>='0' && c<='9') val=val*10+(c-'0');
    else break;
  }

  if(str[i]=='.'){
    i++;
    QuadReal_t exp=1.0;exp/=10;
    for(;str[i]!='\0';i++){
      char c=str[i];
      if(c>='0' && c<='9') val=val+(c-'0')*exp;
      else break;
      exp/=10;
    }
  }

  return sign*val;
}

QuadReal_t fabs(const QuadReal_t f){
  if(f>=0.0) return f;
  else return -f;
}

QuadReal_t sqrt(const QuadReal_t a){
  QuadReal_t b=sqrt((double)a);
  b=(b+a/b)*0.5;
  b=(b+a/b)*0.5;
  return b;
}

QuadReal_t sin(const QuadReal_t a){
  const int N=200;
  static std::vector<QuadReal_t> theta;
  static std::vector<QuadReal_t> sinval;
  static std::vector<QuadReal_t> cosval;
  if(theta.size()==0){
    #pragma omp critical (QUAD_SIN)
    if(theta.size()==0){
      theta.resize(N);
      sinval.resize(N);
      cosval.resize(N);

      QuadReal_t t=1.0;
      for(int i=0;i<N;i++){
        theta[i]=t;
        t=t*0.5;
      }

      sinval[N-1]=theta[N-1];
      cosval[N-1]=1.0-sinval[N-1]*sinval[N-1];
      for(int i=N-2;i>=0;i--){
        sinval[i]=2.0*sinval[i+1]*cosval[i+1];
        cosval[i]=sqrt(1.0-sinval[i]*sinval[i]);
      }
    }
  }

  QuadReal_t t=(a<0.0?-a:a);
  QuadReal_t sval=0.0;
  QuadReal_t cval=1.0;
  for(int i=0;i<N;i++){
    while(theta[i]<=t){
      QuadReal_t sval_=sval*cosval[i]+cval*sinval[i];
      QuadReal_t cval_=cval*cosval[i]-sval*sinval[i];
      sval=sval_;
      cval=cval_;
      t=t-theta[i];
    }
  }
  return (a<0.0?-sval:sval);
}

QuadReal_t cos(const QuadReal_t a){
  const int N=200;
  static std::vector<QuadReal_t> theta;
  static std::vector<QuadReal_t> sinval;
  static std::vector<QuadReal_t> cosval;
  if(theta.size()==0){
    #pragma omp critical (QUAD_COS)
    if(theta.size()==0){
      theta.resize(N);
      sinval.resize(N);
      cosval.resize(N);

      QuadReal_t t=1.0;
      for(int i=0;i<N;i++){
        theta[i]=t;
        t=t*0.5;
      }

      sinval[N-1]=theta[N-1];
      cosval[N-1]=1.0-sinval[N-1]*sinval[N-1];
      for(int i=N-2;i>=0;i--){
        sinval[i]=2.0*sinval[i+1]*cosval[i+1];
        cosval[i]=sqrt(1.0-sinval[i]*sinval[i]);
      }
    }
  }

  QuadReal_t t=(a<0.0?-a:a);
  QuadReal_t sval=0.0;
  QuadReal_t cval=1.0;
  for(int i=0;i<N;i++){
    while(theta[i]<=t){
      QuadReal_t sval_=sval*cosval[i]+cval*sinval[i];
      QuadReal_t cval_=cval*cosval[i]-sval*sinval[i];
      sval=sval_;
      cval=cval_;
      t=t-theta[i];
    }
  }
  return cval;
}

QuadReal_t exp(const QuadReal_t a){
  const int N=200;
  static std::vector<QuadReal_t> theta0;
  static std::vector<QuadReal_t> theta1;
  static std::vector<QuadReal_t> expval0;
  static std::vector<QuadReal_t> expval1;
  if(theta0.size()==0){
    #pragma omp critical (QUAD_EXP)
    if(theta0.size()==0){
      theta0.resize(N);
      theta1.resize(N);
      expval0.resize(N);
      expval1.resize(N);

      theta0[0]=1.0;
      theta1[0]=1.0;
      expval0[0]=const_e<QuadReal_t>();
      expval1[0]=const_e<QuadReal_t>();
      for(int i=1;i<N;i++){
        theta0[i]=theta0[i-1]*0.5;
        theta1[i]=theta1[i-1]*2.0;
        expval0[i]=sqrt(expval0[i-1]);
        expval1[i]=expval1[i-1]*expval1[i-1];
      }
    }
  }

  QuadReal_t t=(a<0.0?-a:a);
  QuadReal_t eval=1.0;
  for(int i=N-1;i>0;i--){
    while(theta1[i]<=t){
      eval=eval*expval1[i];
      t=t-theta1[i];
    }
  }
  for(int i=0;i<N;i++){
    while(theta0[i]<=t){
      eval=eval*expval0[i];
      t=t-theta0[i];
    }
  }
  eval=eval*(1.0+t);
  return (a<0.0?1.0/eval:eval);
}

QuadReal_t log(const QuadReal_t a){
  QuadReal_t y0=log((double)a);
  y0=y0+(a/exp(y0)-1.0);
  y0=y0+(a/exp(y0)-1.0);
  return y0;
}

std::ostream& operator<<(std::ostream& output, const QuadReal_t q_){
  //int width=output.width();
  output<<std::setw(1);

  QuadReal_t q=q_;
  if(q<0.0){
    output<<"-";
    q=-q;
  }else if(q>0){
    output<<" ";
  }else{
    output<<" ";
    output<<"0.0";
    return output;
  }

  int exp=0;
  static const QuadReal_t ONETENTH=(QuadReal_t)1/10;
  while(q<1.0 && abs(exp)<10000){
    q=q*10;
    exp--;
  }
  while(q>=10 && abs(exp)<10000){
    q=q*ONETENTH;
    exp++;
  }

  for(int i=0;i<34;i++){
    output<<(int)q;
    if(i==0) output<<".";
    q=(q-int(q))*10;
    if(q==0 && i>0) break;
  }

  if(exp>0){
    std::cout<<"e+"<<exp;
  }else if(exp<0){
    std::cout<<"e"<<exp;
  }

  return output;
}

