/**
 * \file vector.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains implementation of the class Vector.
 */

#include <cassert>
#include <iostream>
#include <iomanip>

#include <device_wrapper.hpp>

#include <profile.hpp>

namespace pvfmm{

template <class T>
std::ostream& operator<<(std::ostream& output, const Vector<T>& V){
  std::ios::fmtflags f(std::cout.flags());
  output<<std::fixed<<std::setprecision(4)<<std::setiosflags(std::ios::left);
  for(size_t i=0;i<V.Dim();i++)
    output<<std::setw(10)<<V[i]<<' ';
  output<<";\n";
  std::cout.flags(f);
  return output;
}

template <class T>
Vector<T>::Vector(){
  dim=0;
  capacity=0;
  own_data=true;
  data_ptr=sctl::NullIterator<T>();
  dev.dev_ptr=(uintptr_t)NULL;
}

template <class T>
Vector<T>::Vector(size_t dim_, sctl::Iterator<T> data_, bool own_data_){
  dim=dim_;
  capacity=dim;
  own_data=own_data_;
  if(own_data){
    if(dim>0){
      data_ptr=sctl::aligned_new<T>(capacity);
      if(data_!=sctl::NullIterator<T>()) sctl::omp_par::copy((sctl::ConstIterator<T>)data_,(sctl::ConstIterator<T>)data_+(sctl::Long)dim,data_ptr);
    }else data_ptr=sctl::NullIterator<T>();
  }else
    data_ptr=data_;
  dev.dev_ptr=(uintptr_t)NULL;
}

template <class T>
Vector<T>::Vector(const Vector<T>& V){
  dim=V.dim;
  capacity=dim;
  own_data=true;
  if(dim>0){
    data_ptr=sctl::aligned_new<T>(capacity);
    sctl::omp_par::copy((sctl::ConstIterator<T>)V.data_ptr,(sctl::ConstIterator<T>)V.data_ptr+(sctl::Long)dim,data_ptr);
  }else
    data_ptr=sctl::NullIterator<T>();
  dev.dev_ptr=(uintptr_t)NULL;
}

template <class T>
Vector<T>::Vector(const std::vector<T>& V){
  dim=V.size();
  capacity=dim;
  own_data=true;
  if(dim>0){
    data_ptr=sctl::aligned_new<T>(capacity);
    sctl::omp_par::copy(sctl::Ptr2ConstItr<T>(&V[0],dim),sctl::Ptr2ConstItr<T>(&V[0],dim)+(sctl::Long)dim,data_ptr);
  }else
    data_ptr=sctl::NullIterator<T>();
  dev.dev_ptr=(uintptr_t)NULL;
}

template <class T>
Vector<T>::~Vector(){
  FreeDevice(false);
  if(own_data){
    if(data_ptr!=sctl::NullIterator<T>()){
      sctl::aligned_delete<T>(data_ptr);
    }
  }
  data_ptr=sctl::NullIterator<T>();
  capacity=0;
  dim=0;
}

template <class T>
void Vector<T>::Swap(Vector<T>& v1){
  size_t dim_=dim;
  size_t capacity_=capacity;
  sctl::Iterator<T> data_ptr_=data_ptr;
  bool own_data_=own_data;
  Device dev_=dev;

  dim=v1.dim;
  capacity=v1.capacity;
  data_ptr=v1.data_ptr;
  own_data=v1.own_data;
  dev=v1.dev;

  v1.dim=dim_;
  v1.capacity=capacity_;
  v1.data_ptr=data_ptr_;
  v1.own_data=own_data_;
  v1.dev=dev_;
}

template <class T>
void Vector<T>::ReInit(size_t dim_, sctl::Iterator<T> data_, bool own_data_){
  if(own_data_ && own_data && dim_<=capacity){
    if(dim!=dim_) FreeDevice(false);
    dim=dim_;
    if(data_!=sctl::NullIterator<T>()) sctl::omp_par::copy((sctl::ConstIterator<T>)data_,(sctl::ConstIterator<T>)data_+(sctl::Long)dim,data_ptr);
  }else{
    Vector<T> tmp(dim_,data_,own_data_);
    this->Swap(tmp);
  }
}

template <class T>
typename Vector<T>::Device& Vector<T>::AllocDevice(bool copy){
  if(dev.dev_ptr==(uintptr_t)NULL && dim>0) // Allocate data on device.
    dev.dev_ptr=DeviceWrapper::alloc_device((char*)&data_ptr[0], dim*sizeof(T));
  if(dev.dev_ptr!=(uintptr_t)NULL && copy) // Copy data to device
    DeviceWrapper::host2device((char*)&data_ptr[0],(char*)&data_ptr[0],dev.dev_ptr,dim*sizeof(T));

  dev.dim=dim;
  return dev;
}

template <class T>
void Vector<T>::Device2Host(){
  if(dev.dev_ptr==(uintptr_t)NULL) return;
  DeviceWrapper::device2host((char*)&data_ptr[0],dev.dev_ptr,(char*)&data_ptr[0],dim*sizeof(T));
}

template <class T>
void Vector<T>::FreeDevice(bool copy){
  if(dev.dev_ptr==(uintptr_t)NULL) return;
  if(copy) DeviceWrapper::device2host((char*)&data_ptr[0],dev.dev_ptr,(char*)&data_ptr[0],dim*sizeof(T));
  DeviceWrapper::free_device((char*)&data_ptr[0],dev.dev_ptr);
  dev.dev_ptr=(uintptr_t)NULL;
  dev.dim=0;
}

template <class T>
void Vector<T>::Write(const char* fname){
  FILE* f1=fopen(fname,"wb+");
  if(f1==NULL){
    std::cout<<"Unable to open file for writing:"<<fname<<'\n';
    return;
  }
  int dim_=dim;
  fwrite(&dim_,sizeof(int),2,f1);
  if(dim>0) fwrite(&data_ptr[0],sizeof(T),dim,f1);
  fclose(f1);
}

template <class T>
inline size_t Vector<T>::Dim() const{
  return dim;
}

template <class T>
inline size_t Vector<T>::Capacity() const{
  return capacity;
}

template <class T>
void Vector<T>::Resize(size_t dim_){
  if(dim!=dim_) FreeDevice(false);
  if(capacity>=dim_) dim=dim_;
  else ReInit(dim_);
}

template <class T>
void Vector<T>::SetZero(){
  if(dim>0)
    ::memset(&data_ptr[0],0,dim*sizeof(T));
}

template <class ValueType>
ValueType* Vector<ValueType>::Begin(){
  return (dim>0 ? &data_ptr[0] : (ValueType*)NULL);
}

template <class ValueType>
const ValueType* Vector<ValueType>::Begin() const{
  return (dim>0 ? &data_ptr[0] : (const ValueType*)NULL);
}

template <class T>
Vector<T>& Vector<T>::operator=(const Vector<T>& V){
  if(this!=&V){
    if(dim!=V.dim) FreeDevice(false);
    if(capacity<V.dim) ReInit(V.dim);
    dim=V.dim;
    if(dim) sctl::omp_par::copy((sctl::ConstIterator<T>)V.data_ptr,(sctl::ConstIterator<T>)V.data_ptr+(sctl::Long)dim,data_ptr);
  }
  return *this;
}

template <class T>
Vector<T>& Vector<T>::operator=(const std::vector<T>& V){
  {
    if(dim!=V.size()) FreeDevice(false);
    if(capacity<V.size()) ReInit(V.size());
    dim=V.size();
    if (dim) sctl::omp_par::copy(sctl::Ptr2ConstItr<T>(&V[0],dim),sctl::Ptr2ConstItr<T>(&V[0],dim)+(sctl::Long)dim,data_ptr);
  }
  return *this;
}

template <class T>
inline T& Vector<T>::operator[](size_t j){
  assert(j<dim);
  return data_ptr[j];
}

template <class T>
inline const T& Vector<T>::operator[](size_t j) const{
  assert(j<dim);
  return data_ptr[j];
}

}//end namespace
