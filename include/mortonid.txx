/**
 * \file mortonid.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains implementation of the class MortonId.
 */

#include <cmath>
#include <cassert>

namespace pvfmm{

inline MortonId::MortonId():x(0), y(0), z(0), depth(0){}

inline MortonId::MortonId(MortonId m, uint8_t depth_):x(m.x), y(m.y), z(m.z), depth(depth_){
  { // Set depth
    assert(depth<=MAX_DEPTH);
    UINT_T mask=~((((UINT_T)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }
}

template <class T>
inline MortonId::MortonId(T x_f,T y_f, T z_f, uint8_t depth_): depth(depth_){
  static UINT_T max_int=((UINT_T)1)<<(MAX_DEPTH);
  x=(UINT_T)floor(x_f*max_int);
  y=(UINT_T)floor(y_f*max_int);
  z=(UINT_T)floor(z_f*max_int);
  { // Set depth
    assert(depth<=MAX_DEPTH);
    UINT_T mask=~((((UINT_T)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }
}

template <class T>
inline MortonId::MortonId(T* coord, uint8_t depth_): depth(depth_){
  static UINT_T max_int=((UINT_T)1)<<(MAX_DEPTH);
  x=(UINT_T)floor(coord[0]*max_int);
  y=(UINT_T)floor(coord[1]*max_int);
  z=(UINT_T)floor(coord[2]*max_int);
  { // Set depth
    assert(depth<=MAX_DEPTH);
    UINT_T mask=~((((UINT_T)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }
}

template <class T>
inline void MortonId::GetCoord(T* coord){
  static UINT_T max_int=((UINT_T)1)<<(MAX_DEPTH);
  static T s=1.0/((T)max_int);
  coord[0]=x*s;
  coord[1]=y*s;
  coord[2]=z*s;
}

inline unsigned int MortonId::GetDepth() const{
  return depth;
}

inline MortonId MortonId::NextId() const{
  MortonId m=*this;

  UINT_T mask=((UINT_T)1)<<(MAX_DEPTH-depth);
  int i;
  for(i=depth;i>=0;i--){
    m.x=(m.x^mask);
    if((m.x & mask))
      break;
    m.y=(m.y^mask);
    if((m.y & mask))
      break;
    m.z=(m.z^mask);
    if((m.z & mask))
      break;
    mask=(mask<<1);
  }
  if(i<0) i=0;
  m.depth=(uint8_t)i;
  return m;
}

inline MortonId MortonId::getAncestor(uint8_t ancestor_level) const{
  MortonId m=*this;
  m.depth=ancestor_level;

  UINT_T mask=(((UINT_T)1)<<(MAX_DEPTH))-(((UINT_T)1)<<(MAX_DEPTH-ancestor_level));
  m.x=(m.x & mask);
  m.y=(m.y & mask);
  m.z=(m.z & mask);
  return m;
}

inline MortonId MortonId::getDFD(uint8_t level) const{
  MortonId m=*this;
  m.depth=level;
  return m;
}

inline int MortonId::operator<(const MortonId& m) const{
  if(x==m.x && y==m.y && z==m.z) return depth<m.depth;
  UINT_T x_=(x^m.x);
  UINT_T y_=(y^m.y);
  UINT_T z_=(z^m.z);

  if((z_>x_ || ((z_^x_)<x_ && (z_^x_)<z_)) && (z_>y_ || ((z_^y_)<y_ && (z_^y_)<z_)))
    return z<m.z;
  if(y_>x_ || ((y_^x_)<x_ && (y_^x_)<y_))
    return y<m.y;
  return x<m.x;
}

inline int MortonId::operator>(const MortonId& m) const{
  if(x==m.x && y==m.y && z==m.z) return depth>m.depth;
  UINT_T x_=(x^m.x);
  UINT_T y_=(y^m.y);
  UINT_T z_=(z^m.z);

  if((z_>x_ || ((z_^x_)<x_ && (z_^x_)<z_)) && (z_>y_ || ((z_^y_)<y_ && (z_^y_)<z_)))
    return z>m.z;
  if((y_>x_ || ((y_^x_)<x_ && (y_^x_)<y_)))
    return y>m.y;
  return x>m.x;
}

inline int MortonId::operator==(const MortonId& m) const{
  return (x==m.x && y==m.y && z==m.z && depth==m.depth);
}

inline int MortonId::operator!=(const MortonId& m) const{
  return !(*this==m);
}

inline int MortonId::operator<=(const MortonId& m) const{
  return !(*this>m);
}

inline int MortonId::operator>=(const MortonId& m) const{
  return !(*this<m);
}

inline int MortonId::isAncestor(MortonId const & other) const {
  return other.depth>depth && other.getAncestor(depth)==*this;
}

}//end namespace
