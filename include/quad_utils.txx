/**
 * \file quad_utils.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 7-16-2014
 * \brief This file contains quadruple-precision related functions.
 */

#include <iomanip>
#include <cstdlib>
#include <cmath>

QuadReal_t atoquad(const char* str){
  size_t i=0;
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

QuadReal_t fabs(const QuadReal_t& f){
  if(f>=0.0) return f;
  else return -f;
}

QuadReal_t sqrt(const QuadReal_t& a){
  QuadReal_t b=sqrt((double)a);
  b=b+(a/b-b)*0.5;
  b=b+(a/b-b)*0.5;
  return b;
}

QuadReal_t sin(const QuadReal_t& a){
  const size_t N=200;
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

QuadReal_t cos(const QuadReal_t& a){
  const size_t N=200;
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
  return cval;
}

std::ostream& operator<<(std::ostream& output, const QuadReal_t& q_){
  int width=output.width();
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

  for(size_t i=0;i<34;i++){
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

