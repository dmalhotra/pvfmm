/**
 * \file mpi_node.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the class MPI_Node.
 */

#include <cmath>

#include <matrix.hpp>
#include <mem_mgr.hpp>

namespace pvfmm{

template <class T>
MPI_Node<T>::~MPI_Node(){
}

template <class T>
void MPI_Node<T>::Initialize(TreeNode* parent_,int path2node_, TreeNode::NodeData* data_){
  TreeNode::Initialize(parent_,path2node_,data_);

  //Set node coordinates.
  Real_t coord_offset=((Real_t)1.0)/((Real_t)(((uint64_t)1)<<Depth()));
  if(!parent_){
    for(int j=0;j<dim;j++) coord[j]=0;
  }else if(parent_){
    int flag=1;
    for(int j=0;j<dim;j++){
      coord[j]=((MPI_Node<Real_t>*)parent_)->coord[j]+
               ((Path2Node() & flag)?coord_offset:0.0f);
      flag=flag<<1;
    }
  }

  //Initialize colleagues array.
  int n=pvfmm::pow<unsigned int>(3,Dim());
  for(int i=0;i<n;i++) colleague[i]=NULL;

  //Set MPI_Node specific data.
  typename MPI_Node<Real_t>::NodeData* mpi_data=dynamic_cast<typename MPI_Node<Real_t>::NodeData*>(data_);
  if(data_){
    max_pts =mpi_data->max_pts;
    pt_coord=mpi_data->pt_coord;
    pt_value=mpi_data->pt_value;
  }else if(parent){
    max_pts =((MPI_Node<T>*)parent)->max_pts;
    SetGhost(((MPI_Node<T>*)parent)->IsGhost());
  }
}

template <class T>
void MPI_Node<T>::ClearData(){
  pt_coord.ReInit(0);
  pt_value.ReInit(0);
}

template <class T>
MortonId MPI_Node<T>::GetMortonId(){
  assert(coord);
  Real_t s=0.25/(1UL<<MAX_DEPTH);
  return MortonId(coord[0]+s,coord[1]+s,coord[2]+s, Depth()); // TODO: Use interger coordinates instead of floating point.
}

template <class T>
void MPI_Node<T>::SetCoord(MortonId& mid){
  assert(coord);
  mid.GetCoord(coord);
  depth=mid.GetDepth();
}

template <class T>
TreeNode* MPI_Node<T>::NewNode(TreeNode* n_){
  MPI_Node<Real_t>* n=(n_?static_cast<MPI_Node<Real_t>*>(n_):mem::aligned_new<MPI_Node<Real_t> >());
  n->max_pts=max_pts;
  return TreeNode::NewNode(n);
}

template <class T>
bool MPI_Node<T>::SubdivCond(){
  int n=(1UL<<this->Dim());
  // Do not subdivide beyond max_depth
  if(this->Depth()>=this->max_depth-1) return false;
  if(!this->IsLeaf()){ // If has non-leaf children, then return true.
    for(int i=0;i<n;i++){
      MPI_Node<Real_t>* ch=static_cast<MPI_Node<Real_t>*>(this->Child(i));
      assert(ch); //ch==NULL should never happen
      if(!ch->IsLeaf() || ch->IsGhost()) return true;
    }
  }
  else{ // Do not refine ghost leaf nodes.
    if(this->IsGhost()) return false;
  }

  if(!this->IsLeaf()){
    size_t pt_vec_size=0;
    for(int i=0;i<n;i++){
      MPI_Node<Real_t>* ch=static_cast<MPI_Node<Real_t>*>(this->Child(i));
      pt_vec_size+=ch->pt_coord.Dim();
    }
    return pt_vec_size/Dim()>max_pts;
  }else{
    return pt_coord.Dim()/Dim()>max_pts;
  }
}

template <class T>
void MPI_Node<T>::Subdivide(){
  if(!this->IsLeaf()) return;
  TreeNode::Subdivide();
  int nchld=(1UL<<this->Dim());

  if(!IsGhost()){ // Partition point coordinates and values.
    std::vector<Vector<Real_t>*> pt_coord;
    std::vector<Vector<Real_t>*> pt_value;
    std::vector<Vector<size_t>*> pt_scatter;
    this->NodeDataVec(pt_coord, pt_value, pt_scatter);

    std::vector<std::vector<Vector<Real_t>*> > chld_pt_coord(nchld);
    std::vector<std::vector<Vector<Real_t>*> > chld_pt_value(nchld);
    std::vector<std::vector<Vector<size_t>*> > chld_pt_scatter(nchld);
    for(size_t i=0;i<nchld;i++){
      static_cast<MPI_Node<Real_t>*>((MPI_Node<T>*)this->Child(i))
        ->NodeDataVec(chld_pt_coord[i], chld_pt_value[i], chld_pt_scatter[i]);
    }

    Real_t* c=this->Coord();
    Real_t s=pvfmm::pow<Real_t>(0.5,this->Depth()+1);
    for(size_t j=0;j<pt_coord.size();j++){
      if(!pt_coord[j] || !pt_coord[j]->Dim()) continue;
      Vector<Real_t>& coord=*pt_coord[j];
      size_t npts=coord.Dim()/this->dim;

      Vector<size_t> cdata(nchld+1);
      for(size_t i=0;i<nchld+1;i++){
        long long pt1=-1, pt2=npts;
        while(pt2-pt1>1){ // binary search
          long long pt3=(pt1+pt2)/2;
          assert(pt3<npts);
          if(pt3<0) pt3=0;
          int ch_id=(coord[pt3*3+0]>=c[0]+s)*1+
                    (coord[pt3*3+1]>=c[1]+s)*2+
                    (coord[pt3*3+2]>=c[2]+s)*4;
          if(ch_id< i) pt1=pt3;
          if(ch_id>=i) pt2=pt3;
        }
        cdata[i]=pt2;
      }

      if(pt_coord[j]){
        Vector<Real_t>& vec=*pt_coord[j];
        size_t dof=vec.Dim()/npts;
        if(dof>0) for(size_t i=0;i<nchld;i++){
          Vector<Real_t>& chld_vec=*chld_pt_coord[i][j];
          chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
        }
        vec.ReInit(0);
      }
      if(pt_value[j]){
        Vector<Real_t>& vec=*pt_value[j];
        size_t dof=vec.Dim()/npts;
        if(dof>0) for(size_t i=0;i<nchld;i++){
          Vector<Real_t>& chld_vec=*chld_pt_value[i][j];
          chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
        }
        vec.ReInit(0);
      }
      if(pt_scatter[j]){
        Vector<size_t>& vec=*pt_scatter[j];
        size_t dof=vec.Dim()/npts;
        if(dof>0) for(size_t i=0;i<nchld;i++){
          Vector<size_t>& chld_vec=*chld_pt_scatter[i][j];
          chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
        }
        vec.ReInit(0);
      }
    }
  }
};

template <class T>
void MPI_Node<T>::Truncate(){
  if(!this->IsLeaf()){
    int nchld=(1UL<<this->Dim());
    for(size_t i=0;i<nchld;i++){
      if(!this->Child(i)->IsLeaf()){
        this->Child(i)->Truncate();
      }
    }

    std::vector<Vector<Real_t>*> pt_coord;
    std::vector<Vector<Real_t>*> pt_value;
    std::vector<Vector<size_t>*> pt_scatter;
    this->NodeDataVec(pt_coord, pt_value, pt_scatter);

    std::vector<std::vector<Vector<Real_t>*> > chld_pt_coord(nchld);
    std::vector<std::vector<Vector<Real_t>*> > chld_pt_value(nchld);
    std::vector<std::vector<Vector<size_t>*> > chld_pt_scatter(nchld);
    for(size_t i=0;i<nchld;i++){
      static_cast<MPI_Node<Real_t>*>((MPI_Node<T>*)this->Child(i))
        ->NodeDataVec(chld_pt_coord[i], chld_pt_value[i], chld_pt_scatter[i]);
    }

    for(size_t j=0;j<pt_coord.size();j++){
      if(!pt_coord[j]) continue;
      if(pt_coord[j]){
        size_t vec_size=0;
        for(size_t i=0;i<nchld;i++){
          Vector<Real_t>& chld_vec=*chld_pt_coord[i][j];
          vec_size+=chld_vec.Dim();
        }
        Vector<Real_t>& vec=*pt_coord[j];
        vec.ReInit(vec_size);

        vec_size=0;
        for(size_t i=0;i<nchld;i++){
          Vector<Real_t>& chld_vec=*chld_pt_coord[i][j];
          if(chld_vec.Dim()>0){
            mem::memcopy(&vec[vec_size],&chld_vec[0],chld_vec.Dim()*sizeof(Real_t));
            vec_size+=chld_vec.Dim();
          }
        }
      }
      if(pt_value[j]){
        size_t vec_size=0;
        for(size_t i=0;i<nchld;i++){
          Vector<Real_t>& chld_vec=*chld_pt_value[i][j];
          vec_size+=chld_vec.Dim();
        }
        Vector<Real_t>& vec=*pt_value[j];
        vec.ReInit(vec_size);

        vec_size=0;
        for(size_t i=0;i<nchld;i++){
          Vector<Real_t>& chld_vec=*chld_pt_value[i][j];
          if(chld_vec.Dim()>0){
            mem::memcopy(&vec[vec_size],&chld_vec[0],chld_vec.Dim()*sizeof(Real_t));
            vec_size+=chld_vec.Dim();
          }
        }
      }
      if(pt_scatter[j]){
        size_t vec_size=0;
        for(size_t i=0;i<nchld;i++){
          Vector<size_t>& chld_vec=*chld_pt_scatter[i][j];
          vec_size+=chld_vec.Dim();
        }
        Vector<size_t>& vec=*pt_scatter[j];
        vec.ReInit(vec_size);

        vec_size=0;
        for(size_t i=0;i<nchld;i++){
          Vector<size_t>& chld_vec=*chld_pt_scatter[i][j];
          if(chld_vec.Dim()>0){
            mem::memcopy(&vec[vec_size],&chld_vec[0],chld_vec.Dim()*sizeof(Real_t));
            vec_size+=chld_vec.Dim();
          }
        }
      }
    }
  }
  TreeNode::Truncate();
}

template <class T>
PackedData MPI_Node<T>::Pack(bool ghost, void* buff_ptr, size_t offset){
  std::vector<Vector<Real_t>*> pt_coord;
  std::vector<Vector<Real_t>*> pt_value;
  std::vector<Vector<size_t>*> pt_scatter;
  this->NodeDataVec(pt_coord, pt_value, pt_scatter);

  PackedData p0;
  // Determine data length.
  p0.length =sizeof(size_t)+sizeof(int)+sizeof(long long)+sizeof(MortonId);
  for(size_t j=0;j<pt_coord.size();j++){
    p0.length+=3*sizeof(size_t);
    if(pt_coord  [j]) p0.length+=(pt_coord  [j]->Dim())*sizeof(Real_t);
    if(pt_value  [j]) p0.length+=(pt_value  [j]->Dim())*sizeof(Real_t);
    if(pt_scatter[j]) p0.length+=(pt_scatter[j]->Dim())*sizeof(size_t);
  }

  // Allocate memory.
  p0.data=(char*)buff_ptr;
  if(!p0.data){
    this->packed_data.Resize(p0.length+offset);
    p0.data=&this->packed_data[0];
  }

  char* data_ptr=(char*)p0.data;
  data_ptr+=offset;

  // Header
  ((size_t*)data_ptr)[0]=p0.length;
  data_ptr+=sizeof(size_t);

  ((int*)data_ptr)[0]=this->Depth();
  data_ptr+=sizeof(int);

  ((long long*)data_ptr)[0]=this->NodeCost();
  data_ptr+=sizeof(long long);

  ((MortonId*)data_ptr)[0]=this->GetMortonId();
  data_ptr+=sizeof(MortonId);

  // Copy Vector data.
  for(size_t j=0;j<pt_coord.size();j++){
    if(pt_coord[j]){
      Vector<Real_t>& vec=*pt_coord[j];
      ((size_t*)data_ptr)[0]=vec.Dim(); data_ptr+=sizeof(size_t);
      if(vec.Dim()>0 && data_ptr!=(char*)&vec[0])
        mem::memcopy(data_ptr, &vec[0], sizeof(Real_t)*vec.Dim());
      data_ptr+=vec.Dim()*sizeof(Real_t);
    }else{
      ((size_t*)data_ptr)[0]=0; data_ptr+=sizeof(size_t);
    }
    if(pt_value[j]){
      Vector<Real_t>& vec=*pt_value[j];
      ((size_t*)data_ptr)[0]=vec.Dim(); data_ptr+=sizeof(size_t);
      if(vec.Dim()>0 && data_ptr!=(char*)&vec[0])
        mem::memcopy(data_ptr, &vec[0], sizeof(Real_t)*vec.Dim());
      data_ptr+=vec.Dim()*sizeof(Real_t);
    }else{
      ((size_t*)data_ptr)[0]=0; data_ptr+=sizeof(size_t);
    }
    if(pt_scatter[j] && !ghost){
      Vector<size_t>& vec=*pt_scatter[j];
      ((size_t*)data_ptr)[0]=vec.Dim(); data_ptr+=sizeof(size_t);
      if(vec.Dim()>0 && data_ptr!=(char*)&vec[0])
        mem::memcopy(data_ptr, &vec[0], sizeof(size_t)*vec.Dim());
      data_ptr+=vec.Dim()*sizeof(size_t);
    }else{
      ((size_t*)data_ptr)[0]=0; data_ptr+=sizeof(size_t);
    }
  }

  return p0;
}

template <class T>
void MPI_Node<T>::Unpack(PackedData p0, bool own_data){
  std::vector<Vector<Real_t>*> pt_coord;
  std::vector<Vector<Real_t>*> pt_value;
  std::vector<Vector<size_t>*> pt_scatter;
  this->NodeDataVec(pt_coord, pt_value, pt_scatter);

  char* data_ptr=(char*)p0.data;

  // Check header
  assert(((size_t*)data_ptr)[0]==p0.length);
  data_ptr+=sizeof(size_t);

  this->depth=(((int*)data_ptr)[0]);
  data_ptr+=sizeof(int);

  this->NodeCost()=(((long long*)data_ptr)[0]);
  data_ptr+=sizeof(long long);

  this->SetCoord(((MortonId*)data_ptr)[0]);
  data_ptr+=sizeof(MortonId);

  for(size_t j=0;j<pt_coord.size();j++){
    if(pt_coord[j]){
      Vector<Real_t>& vec=*pt_coord[j];
      size_t vec_sz=(((size_t*)data_ptr)[0]); data_ptr+=sizeof(size_t);
      if(own_data){
        vec=Vector<Real_t>(vec_sz,(Real_t*)data_ptr,false);
      }else{
        vec.ReInit(vec_sz,(Real_t*)data_ptr,false);
      }
      data_ptr+=vec_sz*sizeof(Real_t);
    }else{
      assert(!((size_t*)data_ptr)[0]);
      data_ptr+=sizeof(size_t);
    }
    if(pt_value[j]){
      Vector<Real_t>& vec=*pt_value[j];
      size_t vec_sz=(((size_t*)data_ptr)[0]); data_ptr+=sizeof(size_t);
      if(own_data){
        vec=Vector<Real_t>(vec_sz,(Real_t*)data_ptr,false);
      }else{
        vec.ReInit(vec_sz,(Real_t*)data_ptr,false);
      }
      data_ptr+=vec_sz*sizeof(Real_t);
    }else{
      assert(!((size_t*)data_ptr)[0]);
      data_ptr+=sizeof(size_t);
    }
    if(pt_scatter[j]){
      Vector<size_t>& vec=*pt_scatter[j];
      size_t vec_sz=(((size_t*)data_ptr)[0]); data_ptr+=sizeof(size_t);
      if(own_data){
        vec=Vector<size_t>(vec_sz,(size_t*)data_ptr,false);
      }else{
        vec.ReInit(vec_sz,(size_t*)data_ptr,false);
      }
      data_ptr+=vec_sz*sizeof(size_t);
    }else{
      assert(!((size_t*)data_ptr)[0]);
      data_ptr+=sizeof(size_t);
    }
  }
}

template <class T>
void MPI_Node<T>::ReadVal(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, Real_t* val, bool show_ghost){
  if(!pt_coord.Dim()) return;
  size_t n_pts=pt_coord.Dim()/dim;
  size_t data_dof=pt_value.Dim()/n_pts;
  std::vector<Real_t> v(data_dof,0);
  for(size_t i=0;i<n_pts;i++)
    for(int j=0;j<data_dof;j++)
      v[j]+=pt_value[i*data_dof+j];
  for(int j=0;j<data_dof;j++)
    v[j]=v[j]/n_pts;
  for(size_t i=0;i<x.size()*y.size()*z.size()*data_dof;i++){
    val[i]=v[i%data_dof];
  }
}

template <class VTUData_t>
void AppendVTUData(VTUData_t& vtu_data, VTUData_t& new_data){
  typedef typename VTUData_t::VTKReal_t VTKReal_t;

  { // Append new_data to vtk_data
    std::vector<VTKReal_t>&               new_coord=new_data.coord;
    std::vector<std::string>&             new_name =new_data.name;
    std::vector<std::vector<VTKReal_t> >& new_value=new_data.value;

    std::vector<int32_t>& new_connect=new_data.connect;
    std::vector<int32_t>& new_offset =new_data.offset;
    std::vector<uint8_t>& new_types  =new_data.types;



    std::vector<VTKReal_t>&               vtu_coord=vtu_data.coord;
    std::vector<std::string>&             vtu_name =vtu_data.name;
    std::vector<std::vector<VTKReal_t> >& vtu_value=vtu_data.value;

    std::vector<int32_t>& vtu_connect=vtu_data.connect;
    std::vector<int32_t>& vtu_offset =vtu_data.offset;
    std::vector<uint8_t>& vtu_types  =vtu_data.types;



    size_t old_pts=vtu_coord.size()/COORD_DIM;
    size_t new_pts=new_coord.size()/COORD_DIM;

    // New points
    for(size_t i=0;i<new_coord.size();i++){
      vtu_coord.push_back(new_coord[i]);
    }

    // Resize old DataArrays
    for(size_t i=0;i<vtu_value.size();i++){
      size_t curr_size=vtu_value[i].size();
      size_t new_size=(curr_size*(old_pts+new_pts))/old_pts;
      vtu_value[i].resize(new_size,0);
    }

    // Add new DataArrays
    for(size_t i=0;i<new_name.size();i++){
      vtu_name.push_back(new_name[i]);
      vtu_value.push_back(std::vector<VTKReal_t>());
      std::vector<VTKReal_t>& value=vtu_value.back();
      if(new_pts) value.resize((new_value[i].size()*old_pts)/new_pts);
      for(size_t j=0;j<new_value[i].size();j++){
        value.push_back(new_value[i][j]);
      }
    }

    size_t connect_update=old_pts;
    size_t offset_update=vtu_connect.size();

    for(size_t i=0;i<new_connect.size();i++){
      vtu_connect.push_back(connect_update+new_connect[i]);
    }
    for(size_t i=0;i<new_offset.size();i++){
      vtu_offset.push_back(offset_update+new_offset[i]);
    }
    for(size_t i=0;i<new_types.size();i++){
      vtu_types.push_back(new_types[i]);
    }
  }
}

template <class T>
template <class VTUData_t, class Node_t>
void MPI_Node<T>::VTU_Data(VTUData_t& vtu_data, std::vector<Node_t*>& nodes, int lod){
  typedef typename VTUData_t::VTKReal_t VTKReal_t;

  VTUData_t new_data;
  { // Set new data
    new_data.value.resize(1);
    new_data.name.push_back("pt_value");
    std::vector<VTKReal_t>& coord=new_data.coord;
    std::vector<VTKReal_t>& value=new_data.value[0];

    std::vector<int32_t>& connect=new_data.connect;
    std::vector<int32_t>& offset =new_data.offset;
    std::vector<uint8_t>& types  =new_data.types;

    size_t point_cnt=0;
    size_t connect_cnt=0;
    for(size_t nid=0;nid<nodes.size();nid++){
      Node_t* n=nodes[nid];
      if(n->IsGhost() || !n->IsLeaf()) continue;

      for(size_t i=0;i<n->pt_coord.Dim();i+=3){
        coord.push_back(n->pt_coord[i+0]);
        coord.push_back(n->pt_coord[i+1]);
        coord.push_back(n->pt_coord[i+2]);
        connect_cnt++;
        connect.push_back(point_cnt);
        offset.push_back(connect_cnt);
        types.push_back(1);
        point_cnt++;
      }
      for(size_t i=0;i<n->pt_value.Dim();i++){
        value.push_back(n->pt_value[i]);
      }
    }
    size_t value_dof=(value.size()?value.size()/point_cnt:0);
    assert(value_dof*point_cnt==value.size());
    for(size_t nid=0;nid<nodes.size();nid++){
      Node_t* n=nodes[nid];
      if(n->IsGhost() || !n->IsLeaf()) continue;

      Real_t* c=n->Coord();
      Real_t s=pvfmm::pow<Real_t>(0.5,n->Depth());
      for(int i=0;i<8;i++){
        coord.push_back(c[0]+(i&1?1:0)*s);
        coord.push_back(c[1]+(i&2?1:0)*s);
        coord.push_back(c[2]+(i&4?1:0)*s);
        for(int j=0;j<value_dof;j++) value.push_back(0.0);
        connect.push_back(point_cnt+i);
        connect_cnt++;
      }
      offset.push_back(connect_cnt);
      types.push_back(11);
      point_cnt+=8;
    }
  }
  AppendVTUData(vtu_data, new_data);
}

}//end namespace
