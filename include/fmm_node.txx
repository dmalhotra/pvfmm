/**
 * \file fmm_node.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the FMM_Node class.
 */

#include <cassert>

#include <mem_mgr.hpp>
#include <mpi_node.hpp>

namespace pvfmm{

template <class Node>
FMM_Node<Node>::~FMM_Node(){
  if(fmm_data!=NULL) mem::aligned_delete(fmm_data);
  fmm_data=NULL;
}

template <class Node>
void FMM_Node<Node>::Initialize(TreeNode* parent_,int path2node_, TreeNode::NodeData* data_){
  Node::Initialize(parent_,path2node_,data_);

  //Set FMM_Node specific data.
  typename FMM_Node<Node>::NodeData* data=dynamic_cast<typename FMM_Node<Node>::NodeData*>(data_);
  if(data_!=NULL){
    src_coord=data->src_coord;
    src_value=data->src_value;

    surf_coord=data->surf_coord;
    surf_value=data->surf_value;

    trg_coord=data->trg_coord;
    trg_value=data->trg_value;
  }
}


template <class Node>
void FMM_Node<Node>::ClearData(){
  ClearFMMData();
  TreeNode::ClearData();
}


template <class Node>
void FMM_Node<Node>::ClearFMMData(){
  if(fmm_data!=NULL)
    fmm_data->Clear();
}


template <class Node>
TreeNode* FMM_Node<Node>::NewNode(TreeNode* n_){
  FMM_Node<Node>* n=(n_==NULL?mem::aligned_new<FMM_Node<Node> >():static_cast<FMM_Node<Node>*>(n_));
  if(fmm_data!=NULL) n->fmm_data=fmm_data->NewData();
  return Node_t::NewNode(n);
}


template <class Node>
bool FMM_Node<Node>::SubdivCond(){
  int n=(1UL<<this->Dim());
  // Do not subdivide beyond max_depth
  if(this->Depth()>=this->max_depth-1) return false;
  if(!this->IsLeaf()){ // If has non-leaf children, then return true.
    for(int i=0;i<n;i++){
      MPI_Node<Real_t>* ch=static_cast<MPI_Node<Real_t>*>(this->Child(i));
      assert(ch!=NULL); //This should never happen
      if(!ch->IsLeaf() || ch->IsGhost()) return true;
    }
  }
  else{ // Do not refine ghost leaf nodes.
    if(this->IsGhost()) return false;
  }
  if(Node::SubdivCond()) return true;

  if(!this->IsLeaf()){
    size_t pt_vec_size=0;
    for(int i=0;i<n;i++){
      FMM_Node<Node>* ch=static_cast<FMM_Node<Node>*>(this->Child(i));
      pt_vec_size+=ch->src_coord.Dim();
      pt_vec_size+=ch->surf_coord.Dim();
      pt_vec_size+=ch->trg_coord.Dim();
    }
    return pt_vec_size/this->Dim()>this->max_pts;
  }else{
    size_t pt_vec_size=0;
    pt_vec_size+=src_coord.Dim();
    pt_vec_size+=surf_coord.Dim();
    pt_vec_size+=trg_coord.Dim();
    return pt_vec_size/this->Dim()>this->max_pts;
  }
}

template <class Node>
void FMM_Node<Node>::Subdivide(){
  if(!this->IsLeaf()) return;
  Node::Subdivide();
}


template <class Node>
void FMM_Node<Node>::Truncate(){
  Node::Truncate();
}


template <class Node>
PackedData FMM_Node<Node>::Pack(bool ghost, void* buff_ptr, size_t offset){
  PackedData p0,p1,p2;
  if(buff_ptr==NULL){
    p2=PackMultipole();
  }else{
    char* data_ptr=(char*)buff_ptr+offset;
    p2=PackMultipole(data_ptr+2*sizeof(size_t));
  }

  p0.length =sizeof(size_t);
  p0.length+=sizeof(size_t)+p2.length;
  p1=Node_t::Pack(ghost,buff_ptr,p0.length+offset);
  p0.length+=p1.length;
  p0.data=p1.data;

  char* data_ptr=(char*)p0.data;
  data_ptr+=offset;

  // Header
  ((size_t*)data_ptr)[0]=p0.length;
  data_ptr+=sizeof(size_t);

  // Copy multipole data.
  ((size_t*)data_ptr)[0]=p2.length; data_ptr+=sizeof(size_t);
  mem::copy<char>(data_ptr,(char*)p2.data,p2.length);

  return p0;
}

template <class Node>
void FMM_Node<Node>::Unpack(PackedData p0, bool own_data){
  char* data_ptr=(char*)p0.data;

  // Check header
  assert(((size_t*)data_ptr)[0]==p0.length);
  data_ptr+=sizeof(size_t);

  PackedData p2;
  p2.length=(((size_t*)data_ptr)[0]); data_ptr+=sizeof(size_t);
  p2.data=(void*)data_ptr; data_ptr+=p2.length;
  InitMultipole(p2,own_data);

  PackedData p1;
  p1.data=data_ptr;
  p1.length=((size_t*)data_ptr)[0];
  Node::Unpack(p1, own_data);
}


template <class Node>
PackedData FMM_Node<Node>::PackMultipole(void* buff_ptr){
  if(fmm_data!=NULL)
    return fmm_data->PackMultipole(buff_ptr);
  else{
    PackedData pkd;
    pkd.length=0;
    pkd.data=buff_ptr;
    return pkd;
  }
};


template <class Node>
void FMM_Node<Node>::AddMultipole(PackedData data){
  if(data.length>0){
    assert(fmm_data!=NULL);
    fmm_data->AddMultipole(data);
  }
};


template <class Node>
void FMM_Node<Node>::InitMultipole(PackedData data, bool own_data){
  if(data.length>0){
    assert(fmm_data!=NULL);
    fmm_data->InitMultipole(data, own_data);
  }
};

template <class Node>
template <class VTUData_t, class VTUNode_t>
void FMM_Node<Node>::VTU_Data(VTUData_t& vtu_data, std::vector<VTUNode_t*>& nodes, int lod){
  typedef typename VTUData_t::VTKReal_t VTKReal_t;
  Node::VTU_Data(vtu_data, nodes, lod);

  VTUData_t src_data;
  VTUData_t surf_data;
  VTUData_t trg_data;
  { // Set src_data
    VTUData_t& new_data=src_data;
    new_data.value.resize(1);
    new_data.name.push_back("src_value");
    std::vector<VTKReal_t>& coord=new_data.coord;
    std::vector<VTKReal_t>& value=new_data.value[0];

    std::vector<int32_t>& connect=new_data.connect;
    std::vector<int32_t>& offset =new_data.offset;
    std::vector<uint8_t>& types  =new_data.types;

    size_t point_cnt=0;
    size_t connect_cnt=0;
    for(size_t nid=0;nid<nodes.size();nid++){
      VTUNode_t* n=nodes[nid];
      if(n->IsGhost() || !n->IsLeaf()) continue;

      for(size_t i=0;i<n->src_coord.Dim();i+=3){
        coord.push_back(n->src_coord[i+0]);
        coord.push_back(n->src_coord[i+1]);
        coord.push_back(n->src_coord[i+2]);
        connect_cnt++;
        connect.push_back(point_cnt);
        offset.push_back(connect_cnt);
        types.push_back(1);
        point_cnt++;
      }
      for(size_t i=0;i<n->src_value.Dim();i++){
        value.push_back(n->src_value[i]);
      }
    }
  }
  { // Set surf_data
    VTUData_t& new_data=surf_data;
    new_data.value.resize(1);
    new_data.name.push_back("surf_value");
    std::vector<VTKReal_t>& coord=new_data.coord;
    std::vector<VTKReal_t>& value=new_data.value[0];

    std::vector<int32_t>& connect=new_data.connect;
    std::vector<int32_t>& offset =new_data.offset;
    std::vector<uint8_t>& types  =new_data.types;

    size_t point_cnt=0;
    size_t connect_cnt=0;
    for(size_t nid=0;nid<nodes.size();nid++){
      VTUNode_t* n=nodes[nid];
      if(n->IsGhost() || !n->IsLeaf()) continue;

      for(size_t i=0;i<n->surf_coord.Dim();i+=3){
        coord.push_back(n->surf_coord[i+0]);
        coord.push_back(n->surf_coord[i+1]);
        coord.push_back(n->surf_coord[i+2]);
        connect_cnt++;
        connect.push_back(point_cnt);
        offset.push_back(connect_cnt);
        types.push_back(1);
        point_cnt++;
      }
      for(size_t i=0;i<n->surf_value.Dim();i++){
        value.push_back(n->surf_value[i]);
      }
    }
  }
  { // Set trg_data
    VTUData_t& new_data=trg_data;
    new_data.value.resize(1);
    new_data.name.push_back("trg_value");
    std::vector<VTKReal_t>& coord=new_data.coord;
    std::vector<VTKReal_t>& value=new_data.value[0];

    std::vector<int32_t>& connect=new_data.connect;
    std::vector<int32_t>& offset =new_data.offset;
    std::vector<uint8_t>& types  =new_data.types;

    size_t point_cnt=0;
    size_t connect_cnt=0;
    for(size_t nid=0;nid<nodes.size();nid++){
      VTUNode_t* n=nodes[nid];
      if(n->IsGhost() || !n->IsLeaf()) continue;

      for(size_t i=0;i<n->trg_coord.Dim();i+=3){
        coord.push_back(n->trg_coord[i+0]);
        coord.push_back(n->trg_coord[i+1]);
        coord.push_back(n->trg_coord[i+2]);
        connect_cnt++;
        connect.push_back(point_cnt);
        offset.push_back(connect_cnt);
        types.push_back(1);
        point_cnt++;
      }
      for(size_t i=0;i<n->trg_value.Dim();i++){
        value.push_back(n->trg_value[i]);
      }
    }
  }
  AppendVTUData(vtu_data, src_data);
  AppendVTUData(vtu_data, surf_data);
  AppendVTUData(vtu_data, trg_data);
}

}//end namespace
