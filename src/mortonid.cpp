/**
 * \file mortonid.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains implementation of the class MortonId.
 */

#include <mortonid.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

namespace pvfmm{

void MortonId::NbrList(std::vector<MortonId>& nbrs, uint8_t level, int periodic) const{
  nbrs.clear();
  static unsigned int dim=3;
  static unsigned int nbr_cnt=sctl::pow<unsigned int>(3,dim);
  static PVFMM_MID_UINT_T maxCoord=(((PVFMM_MID_UINT_T)1)<<(PVFMM_MAX_DEPTH));

  PVFMM_MID_UINT_T mask=maxCoord-(((PVFMM_MID_UINT_T)1)<<(PVFMM_MAX_DEPTH-level));
  PVFMM_MID_UINT_T pX=x & mask;
  PVFMM_MID_UINT_T pY=y & mask;
  PVFMM_MID_UINT_T pZ=z & mask;

  MortonId mid_tmp;
  mask=(((PVFMM_MID_UINT_T)1)<<(PVFMM_MAX_DEPTH-level));
  for(int i=0; i<nbr_cnt; i++){
    PVFMM_MID_INT_T dX = ((i/1)%3-1)*mask;
    PVFMM_MID_INT_T dY = ((i/3)%3-1)*mask;
    PVFMM_MID_INT_T dZ = ((i/9)%3-1)*mask;
    PVFMM_MID_INT_T newX=(PVFMM_MID_INT_T)pX+dX;
    PVFMM_MID_INT_T newY=(PVFMM_MID_INT_T)pY+dY;
    PVFMM_MID_INT_T newZ=(PVFMM_MID_INT_T)pZ+dZ;
    if(!periodic){
      if(newX>=0 && newX<(PVFMM_MID_INT_T)maxCoord)
      if(newY>=0 && newY<(PVFMM_MID_INT_T)maxCoord)
      if(newZ>=0 && newZ<(PVFMM_MID_INT_T)maxCoord){
        mid_tmp.x=newX; mid_tmp.y=newY; mid_tmp.z=newZ;
        mid_tmp.depth=level;
        nbrs.push_back(mid_tmp);
      }
    }else{
      if(newX<0) newX+=maxCoord; if(newX>=(PVFMM_MID_INT_T)maxCoord) newX-=maxCoord;
      if(newY<0) newY+=maxCoord; if(newY>=(PVFMM_MID_INT_T)maxCoord) newY-=maxCoord;
      if(newZ<0) newZ+=maxCoord; if(newZ>=(PVFMM_MID_INT_T)maxCoord) newZ-=maxCoord;
      mid_tmp.x=newX; mid_tmp.y=newY; mid_tmp.z=newZ;
      mid_tmp.depth=level;
      nbrs.push_back(mid_tmp);
    }
  }
}

std::vector<MortonId> MortonId::Children() const{
  static int dim=3;
  static int c_cnt=(1UL<<dim);
  static PVFMM_MID_UINT_T maxCoord=(((PVFMM_MID_UINT_T)1)<<(PVFMM_MAX_DEPTH));
  std::vector<MortonId> child(c_cnt);

  PVFMM_MID_UINT_T mask=maxCoord-(((PVFMM_MID_UINT_T)1)<<(PVFMM_MAX_DEPTH-depth));
  PVFMM_MID_UINT_T pX=x & mask;
  PVFMM_MID_UINT_T pY=y & mask;
  PVFMM_MID_UINT_T pZ=z & mask;

  mask=(((PVFMM_MID_UINT_T)1)<<(PVFMM_MAX_DEPTH-(depth+1)));
  for(int i=0; i<c_cnt; i++){
    child[i].x=pX+mask*((i/1)%2);
    child[i].y=pY+mask*((i/2)%2);
    child[i].z=pZ+mask*((i/4)%2);
    child[i].depth=(uint8_t)(depth+1);
  }
  return child;
}

std::ostream& operator<<(std::ostream& out, const MortonId & mid){
  double a=0;
  double s=1;
  for(int i=PVFMM_MAX_DEPTH;i>=0;i--){
    s=s*0.5; if(mid.z & (((PVFMM_MID_UINT_T)1)<<i)) a+=s;
    s=s*0.5; if(mid.y & (((PVFMM_MID_UINT_T)1)<<i)) a+=s;
    s=s*0.5; if(mid.x & (((PVFMM_MID_UINT_T)1)<<i)) a+=s;
  }
  out<<"("<<(size_t)mid.x<<","<<(size_t)mid.y<<","<<(size_t)mid.z<<" - "<<a<<")";
  return out;
}

}//end namespace
