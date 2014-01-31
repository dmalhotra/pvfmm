/**
 * \file vector.txx
 * \author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * \date 2-11-2011
 * \brief This file contains implementation of the class Vector.
 */

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iomanip>
#include <profile.hpp>
#include <mem_utils.hpp>
#include <device_wrapper.hpp>

template <class T>
std::ostream& operator<<(std::ostream& output, const Vector<T>& V){
  output<<std::fixed<<std::setprecision(4)<<std::setiosflags(std::ios::left);
  for(size_t i=0;i<V.Dim();i++)
    output<<std::setw(10)<<V[i]<<' ';
  output<<";\n";
  return output;
}

template <class T>
Vector<T>::Vector(){
  dim=0;
  capacity=0;
  own_data=true;
  data_ptr=NULL;
  dev.dev_ptr=(uintptr_t)NULL;
}

template <class T>
Vector<T>::Vector(size_t dim_, T* data_, bool own_data_){
  dim=dim_;
  capacity=dim;
  own_data=own_data_;
  if(own_data){
    if(dim>0){
      data_ptr=mem::aligned_malloc<T>(capacity);
#ifndef __MIC__
      Profile::Add_MEM(capacity*sizeof(T));
#endif
      if(data_!=NULL) mem::memcopy(data_ptr,data_,dim*sizeof(T));
    }else data_ptr=NULL;
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
    data_ptr=mem::aligned_malloc<T>(capacity);
#ifndef __MIC__
    Profile::Add_MEM(capacity*sizeof(T));
#endif
    mem::memcopy(data_ptr,V.data_ptr,dim*sizeof(T));
  }else
    data_ptr=NULL;
  dev.dev_ptr=(uintptr_t)NULL;
}

template <class T>
Vector<T>::Vector(const std::vector<T>& V){
  dim=V.size();
  capacity=dim;
  own_data=true;
  if(dim>0){
    data_ptr=mem::aligned_malloc<T>(capacity);
#ifndef __MIC__
    Profile::Add_MEM(capacity*sizeof(T));
#endif
    mem::memcopy(data_ptr,&V[0],dim*sizeof(T));
  }else
    data_ptr=NULL;
  dev.dev_ptr=(uintptr_t)NULL;
}

template <class T>
Vector<T>::~Vector(){
  FreeDevice(false);
  if(own_data){
    if(data_ptr!=NULL){
      mem::aligned_free(data_ptr);
#ifndef __MIC__
      Profile::Add_MEM(-capacity*sizeof(T));
#endif
    }
  }
  data_ptr=NULL;
  capacity=0;
  dim=0;
}

template <class T>
void Vector<T>::Swap(Vector<T>& v1){
  size_t dim_=dim;
  size_t capacity_=capacity;
  T* data_ptr_=data_ptr;
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
void Vector<T>::ReInit(size_t dim_, T* data_, bool own_data_){
  Vector<T> tmp(dim_,data_,own_data_);
  this->Swap(tmp);
}

template <class T>
typename Vector<T>::Device& Vector<T>::AllocDevice(bool copy){
  if(dev.dev_ptr==(uintptr_t)NULL && dim>0) // Allocate data on device.
    dev.dev_ptr=DeviceWrapper::alloc_device((char*)data_ptr, dim*sizeof(T));
  if(dev.dev_ptr!=(uintptr_t)NULL && copy) // Copy data to device
    DeviceWrapper::host2device((char*)data_ptr,(char*)data_ptr,dev.dev_ptr,dim*sizeof(T));

  dev.dim=dim;
  return dev;
}

template <class T>
void Vector<T>::Device2Host(){
  if(dev.dev_ptr==(uintptr_t)NULL) return;
  DeviceWrapper::device2host((char*)data_ptr,dev.dev_ptr,(char*)data_ptr,dim*sizeof(T));
}

template <class T>
void Vector<T>::FreeDevice(bool copy){
  if(dev.dev_ptr==(uintptr_t)NULL) return;
  if(copy) DeviceWrapper::device2host((char*)data_ptr,dev.dev_ptr,(char*)data_ptr,dim*sizeof(T));
  DeviceWrapper::free_device((char*)data_ptr,dev.dev_ptr);
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
  fwrite(data_ptr,sizeof(T),dim,f1);
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
void Vector<T>::Resize(size_t dim_,bool fit_size){
  ASSERT_WITH_MSG(own_data || capacity>=dim_, "Resizing array beyond capacity when own_data=false.");
  if(dim!=dim_) FreeDevice(false);

  if((capacity>dim_ && !fit_size) || capacity==dim_ || !own_data){
    dim=dim_;
    return;
  }

  {
    if(data_ptr!=NULL){
      mem::aligned_free(data_ptr); data_ptr=NULL;
#ifndef __MIC__
      Profile::Add_MEM(-capacity*sizeof(T));
#endif
    }
    capacity=dim_;
    if(capacity>0){
      data_ptr=mem::aligned_malloc<T>(capacity);
#ifndef __MIC__
      Profile::Add_MEM(capacity*sizeof(T));
#endif
    }
  }
  dim=dim_;
}

template <class T>
void Vector<T>::SetZero(){
  if(dim>0)
    memset(data_ptr,0,dim*sizeof(T));
}

template <class T>
Vector<T>& Vector<T>::operator=(const Vector<T>& V){
  ASSERT_WITH_MSG(own_data || capacity>=V.dim, "Resizing array beyond capacity when own_data=false.");

  if(this!=&V){
    FreeDevice(false);
    if(own_data && capacity<V.dim){
      if(data_ptr!=NULL){
        mem::aligned_free(data_ptr); data_ptr=NULL;
#ifndef __MIC__
        Profile::Add_MEM(-capacity*sizeof(T));
#endif
      }
      capacity=V.dim;
      if(capacity>0){
        data_ptr=mem::aligned_malloc<T>(capacity);
#ifndef __MIC__
        Profile::Add_MEM(capacity*sizeof(T));
#endif
      }
    }
    dim=V.dim;
    mem::memcopy(data_ptr,V.data_ptr,dim*sizeof(T));
  }
  return *this;
}

template <class T>
Vector<T>& Vector<T>::operator=(const std::vector<T>& V){
  ASSERT_WITH_MSG(own_data || capacity>=V.size(), "Resizing array beyond capacity when own_data=false.");

  {
    FreeDevice(false);
    if(own_data && capacity<V.size()){
      if(data_ptr!=NULL){
        mem::aligned_free(data_ptr); data_ptr=NULL;
#ifndef __MIC__
        Profile::Add_MEM(-capacity*sizeof(T));
#endif
      }
      capacity=V.size();
      if(capacity>0){
        data_ptr=mem::aligned_malloc<T>(capacity);
#ifndef __MIC__
        Profile::Add_MEM(capacity*sizeof(T));
#endif
      }
    }
    dim=V.size();
    mem::memcopy(data_ptr,&V[0],dim*sizeof(T));
  }
  return *this;
}

template <class T>
inline T& Vector<T>::operator[](size_t j) const{
  assert(dim>0?j<dim:j==0); //TODO Change to (j<dim)
  return data_ptr[j];
}

