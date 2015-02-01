/**
 * \file cheb_node.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 1-22-2010
 * \brief This file contains the implementation of the class Cheb_Node.
 */

#include <cmath>
#include <cassert>
#include <algorithm>

#include <cheb_utils.hpp>
#include <matrix.hpp>

namespace pvfmm{

template <class Real_t>
Cheb_Node<Real_t>::~Cheb_Node(){}

template <class Real_t>
void Cheb_Node<Real_t>::Initialize(TreeNode* parent_, int path2node_, TreeNode::NodeData* data_) {
  MPI_Node<Real_t>::Initialize(parent_,path2node_,data_);

  //Set Cheb_Node specific data.
  NodeData* cheb_data=dynamic_cast<NodeData*>(data_);
  Cheb_Node<Real_t>* parent=dynamic_cast<Cheb_Node<Real_t>*>(this->Parent());
  if(cheb_data!=NULL){
    cheb_deg=cheb_data->cheb_deg;
    input_fn=cheb_data->input_fn;
    data_dof=cheb_data->data_dof;
    tol=cheb_data->tol;
  }else if(parent!=NULL){
    cheb_deg=parent->cheb_deg;
    input_fn=parent->input_fn;
    data_dof=parent->data_dof;
    tol=parent->tol;
  }

  //Compute Chebyshev approximation.
  if(this->IsLeaf() && !this->IsGhost()){
    if(!input_fn.IsEmpty() && data_dof>0){
      Real_t s=pow(0.5,this->Depth());
      int n1=(int)(pow((Real_t)(cheb_deg+1),this->Dim())+0.5);
      std::vector<Real_t> coord=cheb_nodes<Real_t>(cheb_deg,this->Dim());
      for(int i=0;i<n1;i++){
        coord[i*3+0]=coord[i*3+0]*s+this->Coord()[0];
        coord[i*3+1]=coord[i*3+1]*s+this->Coord()[1];
        coord[i*3+2]=coord[i*3+2]*s+this->Coord()[2];
      }

      std::vector<Real_t> input_val(n1*data_dof);
      input_fn(&coord[0],n1,&input_val[0]);
      Matrix<Real_t> M_val(n1,data_dof,&input_val[0],false);
      M_val=M_val.Transpose();

      cheb_coeff.Resize(((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6*data_dof); cheb_coeff.SetZero();
      cheb_approx<Real_t,Real_t>(&input_val[0], cheb_deg, data_dof, &cheb_coeff[0]);
    }else if(this->cheb_value.Dim()>0){
      size_t n_ptr=this->cheb_coord.Dim()/this->Dim();
      assert(n_ptr*data_dof==this->cheb_value.Dim());
      points2cheb<Real_t>(cheb_deg,&(this->cheb_coord[0]),&(this->cheb_value[0]),
          this->cheb_coord.Dim()/this->Dim(),data_dof,this->Coord(),
          1.0/(1UL<<this->Depth()), cheb_coeff);
    }
  }
}

template <class Real_t>
void Cheb_Node<Real_t>::ClearData(){
  ChebData().Resize(0);
  MPI_Node<Real_t>::ClearData();
}

template <class Real_t>
TreeNode* Cheb_Node<Real_t>::NewNode(TreeNode* n_){
  Cheb_Node<Real_t>* n=(n_==NULL?mem::aligned_new<Cheb_Node<Real_t> >():static_cast<Cheb_Node<Real_t>*>(n_));
  n->cheb_deg=cheb_deg;
  n->input_fn=input_fn;
  n->data_dof=data_dof;
  n->tol=tol;
  return MPI_Node<Real_t>::NewNode(n);
}

template <class Real_t>
bool Cheb_Node<Real_t>::SubdivCond(){
  // Do not subdivide beyond max_depth
  if(this->Depth()>=this->max_depth-1) return false;
  if(!this->IsLeaf()){ // If has non-leaf children, then return true.
    int n=(1UL<<this->Dim());
    for(int i=0;i<n;i++){
      Cheb_Node<Real_t>* ch=static_cast<Cheb_Node<Real_t>*>(this->Child(i));
      assert(ch!=NULL); //This should never happen
      if(!ch->IsLeaf() || ch->IsGhost()) return true;
    }
  }
  else{ // Do not refine ghost leaf nodes.
    if(this->IsGhost()) return false;
  }
  if(MPI_Node<Real_t>::SubdivCond()) return true;

  if(!this->IsLeaf()){
    std::vector<Real_t> val((cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1)*data_dof);
    std::vector<Real_t> x=cheb_nodes<Real_t>(cheb_deg,1);
    std::vector<Real_t> y=x;
    std::vector<Real_t> z=x;
    Real_t s=pow(0.5,this->Depth());
    Real_t* coord=this->Coord();
    for(size_t i=0;i<x.size();i++){
      x[i]=x[i]*s+coord[0];
      y[i]=y[i]*s+coord[1];
      z[i]=z[i]*s+coord[2];
    }
    read_val(x,y,z, cheb_deg+1, cheb_deg+1, cheb_deg+1, &val[0]);

    std::vector<Real_t> tmp_coeff(data_dof*((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
    cheb_approx<Real_t,Real_t>(&val[0],cheb_deg,data_dof,&tmp_coeff[0]);
    return (cheb_err(&(tmp_coeff[0]),cheb_deg,data_dof)>tol);
  }else{
    assert(cheb_coeff.Dim()>0);
    Real_t err=cheb_err(&(cheb_coeff[0]),cheb_deg,data_dof);
    return (err>tol);
  }
}

template <class Real_t>
void Cheb_Node<Real_t>::Subdivide() {
  if(!this->IsLeaf()) return;
  MPI_Node<Real_t>::Subdivide();
  if(cheb_deg<0 || cheb_coeff.Dim()==0 || !input_fn.IsEmpty()) return;

  std::vector<Real_t> x(cheb_deg+1);
  std::vector<Real_t> y(cheb_deg+1);
  std::vector<Real_t> z(cheb_deg+1);
  Vector<Real_t> cheb_node=cheb_nodes<Real_t>(cheb_deg,1);
  Vector<Real_t> val((size_t)pow(cheb_deg+1,this->Dim())*data_dof);
  Vector<Real_t> child_cheb_coeff[8];
  int n=(1UL<<this->Dim());
  for(int i=0;i<n;i++){
    Real_t coord[3]={(Real_t)((i  )%2?0:-1.0),
                     (Real_t)((i/2)%2?0:-1.0),
                     (Real_t)((i/4)%2?0:-1.0)};
    for(int j=0;j<=cheb_deg;j++){
      x[j]=cheb_node[j]+coord[0];
      y[j]=cheb_node[j]+coord[1];
      z[j]=cheb_node[j]+coord[2];
    }
    cheb_eval(cheb_coeff, cheb_deg, x, y, z, val);
    assert(val.Dim()==pow(cheb_deg+1,this->Dim())*data_dof);
    child_cheb_coeff[i].Resize(data_dof*((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
    cheb_approx<Real_t,Real_t>(&val[0],cheb_deg,data_dof,&(child_cheb_coeff[i][0]));

    Cheb_Node<Real_t>* child=static_cast<Cheb_Node<Real_t>*>(this->Child(i));
    child->cheb_coeff=child_cheb_coeff[i];
    assert(child->cheb_deg==cheb_deg);
    assert(child->tol==tol);
  }
}

template <class Real_t>
void Cheb_Node<Real_t>::Truncate() {
  if(cheb_deg<0 || this->IsLeaf())
    return MPI_Node<Real_t>::Truncate();

  std::vector<Real_t> cheb_node=cheb_nodes<Real_t>(cheb_deg,1);
  std::vector<Real_t> x=cheb_node;
  std::vector<Real_t> y=cheb_node;
  std::vector<Real_t> z=cheb_node;
  std::vector<Real_t> val((cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1)*data_dof);
  Real_t s=pow(0.5,this->Depth());
  Real_t* coord=this->Coord();
  for(size_t i=0;i<x.size();i++){
    x[i]=x[i]*s+coord[0];
    y[i]=y[i]*s+coord[1];
    z[i]=z[i]*s+coord[2];
  }
  read_val(x,y,z, cheb_deg+1, cheb_deg+1, cheb_deg+1, &val[0]);
  cheb_coeff.Resize(data_dof*((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
  cheb_approx<Real_t,Real_t>(&val[0],cheb_deg,data_dof,&cheb_coeff[0]);
  MPI_Node<Real_t>::Truncate();
}

template <class Real_t>
PackedData Cheb_Node<Real_t>::Pack(bool ghost, void* buff_ptr, size_t offset) {
  return MPI_Node<Real_t>::Pack(ghost,buff_ptr, offset);
}

template <class Real_t>
void Cheb_Node<Real_t>::Unpack(PackedData p0, bool own_data) {
  MPI_Node<Real_t>::Unpack(p0, own_data);
}

template <class Real_t>
template <class VTUData_t, class Node_t>
void Cheb_Node<Real_t>::VTU_Data(VTUData_t& vtu_data, std::vector<Node_t*>& nodes, int lod){
  typedef typename VTUData_t::VTKReal_t VTKReal_t;
  //MPI_Node<Real_t>::VTU_Data(vtu_data, nodes, lod);

  VTUData_t new_data;
  { // Set new data
    new_data.value.resize(1);
    new_data.name.push_back("cheb_value");
    std::vector<VTKReal_t>& coord=new_data.coord;
    std::vector<VTKReal_t>& value=new_data.value[0];

    std::vector<int32_t>& connect=new_data.connect;
    std::vector<int32_t>& offset =new_data.offset;
    std::vector<uint8_t>& types  =new_data.types;

    int gridpt_cnt;
    std::vector<Real_t> grid_pts;
    {
      int cheb_deg=lod-1;
      std::vector<Real_t> cheb_node_;
      if(cheb_deg>=0) cheb_node_=cheb_nodes<Real_t>(cheb_deg,1);
      gridpt_cnt=cheb_node_.size()+2;
      grid_pts.resize(gridpt_cnt);
      for(size_t i=0;i<cheb_node_.size();i++) grid_pts[i+1]=cheb_node_[i];
      grid_pts[0]=0.0; grid_pts[gridpt_cnt-1]=1.0;
    }

    Vector<Real_t> gridval;
    for(size_t nid=0;nid<nodes.size();nid++){
      Node_t* n=nodes[nid];
      if(n->IsGhost() || !n->IsLeaf()) continue;

      size_t point_cnt=coord.size()/COORD_DIM;
      Real_t* c=n->Coord();
      Real_t s=pow(0.5,n->Depth());
      for(int i0=0;i0<gridpt_cnt;i0++)
      for(int i1=0;i1<gridpt_cnt;i1++)
      for(int i2=0;i2<gridpt_cnt;i2++){
        coord.push_back(c[0]+grid_pts[i2]*s);
        coord.push_back(c[1]+grid_pts[i1]*s);
        coord.push_back(c[2]+grid_pts[i0]*s);
        for(int j=0;j<n->data_dof;j++) value.push_back(0.0);
      }
      for(int i0=0;i0<(gridpt_cnt-1);i0++)
      for(int i1=0;i1<(gridpt_cnt-1);i1++)
      for(int i2=0;i2<(gridpt_cnt-1);i2++){
        for(int j0=0;j0<2;j0++)
        for(int j1=0;j1<2;j1++)
        for(int j2=0;j2<2;j2++)
          connect.push_back(point_cnt + ((i0+j0)*gridpt_cnt + (i1+j1))*gridpt_cnt + (i2+j2)*1);
        offset.push_back(connect.size());
        types.push_back(11);
      }

      {// Set point values.
        if(gridval.Dim()!=n->data_dof*gridpt_cnt*gridpt_cnt*gridpt_cnt)
          gridval.ReInit(n->data_dof*gridpt_cnt*gridpt_cnt*gridpt_cnt);
        std::vector<Real_t> x(gridpt_cnt);
        std::vector<Real_t> y(gridpt_cnt);
        std::vector<Real_t> z(gridpt_cnt);
        for(int i=0;i<gridpt_cnt;i++){
          x[i]=c[0]+s*grid_pts[i];
          y[i]=c[1]+s*grid_pts[i];
          z[i]=c[2]+s*grid_pts[i];
        }
        n->ReadVal(x, y, z, &gridval[0]);
        //Rearrrange data
        //(x1,x2,x3,...,y1,y2,...z1,...) => (x1,y1,z1,x2,y2,z2,...)
        Matrix<VTKReal_t> M(n->data_dof,gridpt_cnt*gridpt_cnt*gridpt_cnt,&value[point_cnt*n->data_dof],false);
        for(size_t i=0;i<gridval.Dim();i++) M[0][i]=gridval[i];
        M=M.Transpose();
      }
    }
  }
  AppendVTUData(vtu_data, new_data);
}

template <class Real_t>
void Cheb_Node<Real_t>::Gradient(){
  int dim=3;//this->Dim();
  if(this->IsLeaf() && ChebData().Dim()>0){
    Real_t scale=pow(2,this->depth);
    for(size_t i=0;i<ChebData().Dim();i++)
      ChebData()[i]*=scale;

    Vector<Real_t> coeff(ChebData().Dim()*dim);
    cheb_grad(ChebData(),cheb_deg,coeff);
    ChebData().Swap(coeff);
  }
  data_dof*=3;
}

template <class Real_t>
void Cheb_Node<Real_t>::Divergence(){
  int dim=3;//this->Dim();
  if(this->IsLeaf() && ChebData().Dim()>0){
    assert(data_dof%3==0);
    int n3=((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6;
    Vector<Real_t> coeff(ChebData().Dim()/dim);
    for(int i=0;i<data_dof;i=i+dim)
      cheb_div(&(ChebData()[n3*i]),cheb_deg,&coeff[n3*(i/dim)]);
    ChebData().Swap(coeff);

    Real_t scale=pow(2,this->depth);
    for(size_t i=0;i<ChebData().Dim();i++)
      ChebData()[i]*=scale;
  }
  data_dof/=dim;
}

template <class Real_t>
void Cheb_Node<Real_t>::Curl(){
  int dim=3;//this->Dim();
  if(this->IsLeaf() && ChebData().Dim()>0){
    assert(data_dof%dim==0);
    int n3=((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6;
    Vector<Real_t> coeff(ChebData().Dim());
    for(int i=0;i<data_dof;i=i+dim)
      cheb_curl(&(ChebData()[n3*i]),cheb_deg,&coeff[n3*i]);
    ChebData().Swap(coeff);

    Real_t scale=pow(2,this->depth);
    for(size_t i=0;i<ChebData().Dim();i++)
      ChebData()[i]*=scale;
  }
}

template <class Real_t>
void Cheb_Node<Real_t>::read_val(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, int nx, int ny, int nz, Real_t* val, bool show_ghost/*=true*/){
  if(cheb_deg<0) return;
  Real_t s=0.5*pow(0.5,this->Depth());
  Real_t s_inv=1/s;
  if(this->IsLeaf()){
    if(cheb_coeff.Dim()!=(size_t)((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6*data_dof
        || (this->IsGhost() && !show_ghost)) return; Vector<Real_t> out;
    std::vector<Real_t> x_=x;
    std::vector<Real_t> y_=y;
    std::vector<Real_t> z_=z;
    for(size_t i=0;i<x.size();i++)
      x_[i]=(x[i]-this->Coord()[0])*s_inv-1.0;
    for(size_t i=0;i<y.size();i++)
      y_[i]=(y[i]-this->Coord()[1])*s_inv-1.0;
    for(size_t i=0;i<z.size();i++)
      z_[i]=(z[i]-this->Coord()[2])*s_inv-1.0;
    cheb_eval(cheb_coeff, cheb_deg, x_, y_, z_, out);

    for(int l=0;l<data_dof;l++)
    for(size_t i=0;i<x.size();i++)
    for(size_t j=0;j<y.size();j++)
    for(size_t k=0;k<z.size();k++){
      val[i+(j+(k+l*nz)*ny)*nx]=out[i+(j+(k+l*z.size())*y.size())*x.size()];
    }
    return;
  }
  Real_t coord_[3]={this->Coord()[0]+s, this->Coord()[1]+s, this->Coord()[2]+s};
  Real_t* indx[3]={&(std::lower_bound(x.begin(),x.end(),coord_[0])[0]),
    &(std::lower_bound(y.begin(),y.end(),coord_[1])[0]),
    &(std::lower_bound(z.begin(),z.end(),coord_[2])[0])};
  std::vector<Real_t> x1[2]={std::vector<Real_t>(&(x.begin()[0]),indx[0]),std::vector<Real_t>(indx[0],&(x.end()[0]))};
  std::vector<Real_t> y1[2]={std::vector<Real_t>(&(y.begin()[0]),indx[1]),std::vector<Real_t>(indx[1],&(y.end()[0]))};
  std::vector<Real_t> z1[2]={std::vector<Real_t>(&(z.begin()[0]),indx[2]),std::vector<Real_t>(indx[2],&(z.end()[0]))};

  for(int i=0;i<8;i++){
    std::vector<Real_t>& x1_=x1[i%2];
    std::vector<Real_t>& y1_=y1[(i/2)%2];
    std::vector<Real_t>& z1_=z1[(i/4)%2];
    if(x1_.size()>0 && y1_.size()>0 && z1_.size()>0){
      static_cast<Cheb_Node<Real_t>*>(this->Child(i))->read_val(x1_,y1_,z1_,nx,ny,nz,&val[(i%2?indx[0]-&x[0]:0)+((i/2)%2?indx[1]-&y[0]:0)*nx+((i/4)%2?indx[2]-&z[0]:0)*nx*ny],show_ghost);
    }
  }
}

}//end namespace
