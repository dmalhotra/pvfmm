/**
 * \file fmm_pts.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 3-07-2011
 * \brief This file contains the implementation of the FMM_Pts class.
 */

#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <iostream>
#include <stdint.h>
#include <set>
#ifdef PVFMM_HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#if defined(__MIC__)
#include <immintrin.h>
#endif

#include <profile.hpp>
#include <cheb_utils.hpp>

namespace pvfmm{

/**
 * \brief Returns the coordinates of points on the surface of a cube.
 * \param[in] p Number of points on an edge of the cube is (n+1)
 * \param[in] c Coordinates to the centre of the cube (3D array).
 * \param[in] alpha Scaling factor for the size of the cube.
 * \param[in] depth Depth of the cube in the octree.
 * \return Vector with coordinates of points on the surface of the cube in the
 * format [x0 y0 z0 x1 y1 z1 .... ].
 */
template <class Real_t>
std::vector<Real_t> surface(int p, Real_t* c, Real_t alpha, int depth){
  size_t n_=(6*(p-1)*(p-1)+2);  //Total number of points.

  std::vector<Real_t> coord(n_*3);
  coord[0]=coord[1]=coord[2]=-1.0;
  size_t cnt=1;
  for(int i=0;i<p-1;i++)
    for(int j=0;j<p-1;j++){
      coord[cnt*3  ]=-1.0;
      coord[cnt*3+1]=(2.0*(i+1)-p+1)/(p-1);
      coord[cnt*3+2]=(2.0*j-p+1)/(p-1);
      cnt++;
    }
  for(int i=0;i<p-1;i++)
    for(int j=0;j<p-1;j++){
      coord[cnt*3  ]=(2.0*i-p+1)/(p-1);
      coord[cnt*3+1]=-1.0;
      coord[cnt*3+2]=(2.0*(j+1)-p+1)/(p-1);
      cnt++;
    }
  for(int i=0;i<p-1;i++)
    for(int j=0;j<p-1;j++){
      coord[cnt*3  ]=(2.0*(i+1)-p+1)/(p-1);
      coord[cnt*3+1]=(2.0*j-p+1)/(p-1);
      coord[cnt*3+2]=-1.0;
      cnt++;
    }
  for(size_t i=0;i<(n_/2)*3;i++)
    coord[cnt*3+i]=-coord[i];

  Real_t r = 0.5*pvfmm::pow<Real_t>(0.5,depth);
  Real_t b = alpha*r;
  for(size_t i=0;i<n_;i++){
    coord[i*3+0]=(coord[i*3+0]+1.0)*b+c[0];
    coord[i*3+1]=(coord[i*3+1]+1.0)*b+c[1];
    coord[i*3+2]=(coord[i*3+2]+1.0)*b+c[2];
  }
  return coord;
}

/**
 * \brief Returns the coordinates of points on the upward check surface of cube.
 * \see surface()
 */
template <class Real_t>
std::vector<Real_t> u_check_surf(int p, Real_t* c, int depth){
  Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
  Real_t coord[3]={(Real_t)(c[0]-r*(RAD1-1.0)),(Real_t)(c[1]-r*(RAD1-1.0)),(Real_t)(c[2]-r*(RAD1-1.0))};
  return surface(p,coord,(Real_t)RAD1,depth);
}

/**
 * \brief Returns the coordinates of points on the upward equivalent surface of cube.
 * \see surface()
 */
template <class Real_t>
std::vector<Real_t> u_equiv_surf(int p, Real_t* c, int depth){
  Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
  Real_t coord[3]={(Real_t)(c[0]-r*(RAD0-1.0)),(Real_t)(c[1]-r*(RAD0-1.0)),(Real_t)(c[2]-r*(RAD0-1.0))};
  return surface(p,coord,(Real_t)RAD0,depth);
}

/**
 * \brief Returns the coordinates of points on the downward check surface of cube.
 * \see surface()
 */
template <class Real_t>
std::vector<Real_t> d_check_surf(int p, Real_t* c, int depth){
  Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
  Real_t coord[3]={(Real_t)(c[0]-r*(RAD0-1.0)),(Real_t)(c[1]-r*(RAD0-1.0)),(Real_t)(c[2]-r*(RAD0-1.0))};
  return surface(p,coord,(Real_t)RAD0,depth);
}

/**
 * \brief Returns the coordinates of points on the downward equivalent surface of cube.
 * \see surface()
 */
template <class Real_t>
std::vector<Real_t> d_equiv_surf(int p, Real_t* c, int depth){
  Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
  Real_t coord[3]={(Real_t)(c[0]-r*(RAD1-1.0)),(Real_t)(c[1]-r*(RAD1-1.0)),(Real_t)(c[2]-r*(RAD1-1.0))};
  return surface(p,coord,(Real_t)RAD1,depth);
}

/**
 * \brief Defines the 3D grid for convolution in FFT acceleration of V-list.
 * \see surface()
 */
template <class Real_t>
std::vector<Real_t> conv_grid(int p, Real_t* c, int depth){
  Real_t r=pvfmm::pow<Real_t>(0.5,depth);
  Real_t a=r*RAD0;
  Real_t coord[3]={c[0],c[1],c[2]};
  int n1=p*2;
  int n2=pvfmm::pow<int>((Real_t)n1,2);
  int n3=pvfmm::pow<int>((Real_t)n1,3);
  std::vector<Real_t> grid(n3*3);
  for(int i=0;i<n1;i++)
  for(int j=0;j<n1;j++)
  for(int k=0;k<n1;k++){
    grid[(i+n1*j+n2*k)*3+0]=(i-p)*a/(p-1)+coord[0];
    grid[(i+n1*j+n2*k)*3+1]=(j-p)*a/(p-1)+coord[1];
    grid[(i+n1*j+n2*k)*3+2]=(k-p)*a/(p-1)+coord[2];
  }
  return grid;
}

template <class Real_t>
void FMM_Data<Real_t>::Clear(){
  upward_equiv.Resize(0);
}

template <class Real_t>
PackedData FMM_Data<Real_t>::PackMultipole(void* buff_ptr){
  PackedData p0; p0.data=buff_ptr;
  p0.length=upward_equiv.Dim()*sizeof(Real_t);
  if(p0.length==0) return p0;

  if(p0.data==NULL) p0.data=(char*)&upward_equiv[0];
  else mem::memcopy(p0.data,&upward_equiv[0],p0.length);
  return p0;
}

template <class Real_t>
void FMM_Data<Real_t>::AddMultipole(PackedData p0){
  Real_t* data=(Real_t*)p0.data;
  size_t n=p0.length/sizeof(Real_t);
  assert(upward_equiv.Dim()==n);
  Matrix<Real_t> v0(1,n,&upward_equiv[0],false);
  Matrix<Real_t> v1(1,n,data,false);
  v0+=v1;
}

template <class Real_t>
void FMM_Data<Real_t>::InitMultipole(PackedData p0, bool own_data){
  Real_t* data=(Real_t*)p0.data;
  size_t n=p0.length/sizeof(Real_t);
  if(n==0) return;
  if(own_data){
    upward_equiv=Vector<Real_t>(n, &data[0], false);
  }else{
    upward_equiv.ReInit(n, &data[0], false);
  }
}

template <class FMMNode>
FMM_Pts<FMMNode>::~FMM_Pts() {
  if(mat!=NULL){
//    int rank;
//    MPI_Comm_rank(comm,&rank);
//    if(rank==0) mat->Save2File("Precomp.data");
    delete mat;
    mat=NULL;
  }
  if(vprecomp_fft_flag) FFTW_t<Real_t>::fft_destroy_plan(vprecomp_fftplan);
  #ifdef __INTEL_OFFLOAD0
  #pragma offload target(mic:0)
  #endif
  {
    if(vlist_fft_flag ) FFTW_t<Real_t>::fft_destroy_plan(vlist_fftplan );
    if(vlist_ifft_flag) FFTW_t<Real_t>::fft_destroy_plan(vlist_ifftplan);
    vlist_fft_flag =false;
    vlist_ifft_flag=false;
  }
}



template <class FMMNode>
void FMM_Pts<FMMNode>::Initialize(int mult_order, const MPI_Comm& comm_, const Kernel<Real_t>* kernel_){
  Profile::Tic("InitFMM_Pts",&comm_,true);{

  int rank;
  MPI_Comm_rank(comm_,&rank);
  bool verbose=false;
  #ifndef NDEBUG
  #ifdef __VERBOSE__
  if(!rank) verbose=true;
  #endif
  #endif
  if(kernel_) kernel_->Initialize(verbose);

  multipole_order=mult_order;
  comm=comm_;
  kernel=kernel_;
  assert(kernel!=NULL);

  bool save_precomp=false;
  mat=new PrecompMat<Real_t>(ScaleInvar());
  if(this->mat_fname.size()==0){// && !this->ScaleInvar()){
    std::stringstream st;
    st<<PVFMM_PRECOMP_DATA_PATH;

    if(!st.str().size()){ // look in PVFMM_DIR
      char* pvfmm_dir = getenv ("PVFMM_DIR");
      if(pvfmm_dir) st<<pvfmm_dir;
    }

    #ifndef STAT_MACROS_BROKEN
    if(st.str().size()){ // check if the path is a directory
      struct stat stat_buff;
      if(stat(st.str().c_str(), &stat_buff) || !S_ISDIR(stat_buff.st_mode)){
        std::cout<<"error: path not found: "<<st.str()<<'\n';
        exit(0);
      }
    }
    #endif

    if(st.str().size()) st<<'/';
    st<<"Precomp_"<<kernel->ker_name.c_str()<<"_m"<<mult_order;
    if(sizeof(Real_t)==8) st<<"";
    else if(sizeof(Real_t)==4) st<<"_f";
    else st<<"_t"<<sizeof(Real_t);
    {
        switch(periodicType){
        case PeriodicType::NONE :
        	st<<"PNONE";
      	  break;
        case PeriodicType::PXYZ :
        	st<<"PXYZ";
      	  break;
        case PeriodicType::PZ :
        	st<<"PZ";
      	  break;
        case PeriodicType::PXY :
        	st<<"PXY";
      	  break;
        }
    }
    st<<".data";
    this->mat_fname=st.str();
    save_precomp=true;
  }
  this->mat->LoadFile(mat_fname.c_str(), this->comm);

  interac_list.Initialize(COORD_DIM, this->mat);

  Profile::Tic("PrecompUC2UE",&comm,false,4);
  this->PrecompAll(UC2UE0_Type);
  this->PrecompAll(UC2UE1_Type);
  Profile::Toc();

  Profile::Tic("PrecompDC2DE",&comm,false,4);
  this->PrecompAll(DC2DE0_Type);
  this->PrecompAll(DC2DE1_Type);
  Profile::Toc();

  Profile::Tic("PrecompBC",&comm,false,4);
  { /*
    int type=BC_Type;
    for(int l=0;l<MAX_DEPTH;l++)
    for(size_t indx=0;indx<this->interac_list.ListCount((Mat_Type)type);indx++){
      Matrix<Real_t>& M=this->mat->Mat(l, (Mat_Type)type, indx);
      M.Resize(0,0);
    } // */
  }
  this->PrecompAll(BC_Type,0);
  Profile::Toc();

  Profile::Tic("PrecompU2U",&comm,false,4);
  this->PrecompAll(U2U_Type);
  Profile::Toc();

  Profile::Tic("PrecompD2D",&comm,false,4);
  this->PrecompAll(D2D_Type);
  Profile::Toc();

  if(save_precomp){
    Profile::Tic("Save2File",&this->comm,false,4);
    if(!rank){
      FILE* f=fopen(this->mat_fname.c_str(),"r");
      if(f==NULL) { //File does not exists.
        this->mat->Save2File(this->mat_fname.c_str());
      }else fclose(f);
    }
    Profile::Toc();
  }

  Profile::Tic("PrecompV",&comm,false,4);
  this->PrecompAll(V_Type);
  Profile::Toc();
  Profile::Tic("PrecompV1",&comm,false,4);
  this->PrecompAll(V1_Type);
  Profile::Toc();

  }Profile::Toc();
}

template <class Real_t>
Permutation<Real_t> equiv_surf_perm(size_t m, size_t p_indx, const Permutation<Real_t>& ker_perm, const Vector<Real_t>* scal_exp=NULL){
  Real_t eps=1e-10;
  int dof=ker_perm.Dim();

  Real_t c[3]={-0.5,-0.5,-0.5};
  std::vector<Real_t> trg_coord=d_check_surf(m,c,0);
  int n_trg=trg_coord.size()/3;

  Permutation<Real_t> P=Permutation<Real_t>(n_trg*dof);
  if(p_indx==ReflecX || p_indx==ReflecY || p_indx==ReflecZ){ // Set P.perm
    for(int i=0;i<n_trg;i++)
    for(int j=0;j<n_trg;j++){
      if(pvfmm::fabs<Real_t>(trg_coord[i*3+0]-trg_coord[j*3+0]*(p_indx==ReflecX?-1.0:1.0))<eps)
      if(pvfmm::fabs<Real_t>(trg_coord[i*3+1]-trg_coord[j*3+1]*(p_indx==ReflecY?-1.0:1.0))<eps)
      if(pvfmm::fabs<Real_t>(trg_coord[i*3+2]-trg_coord[j*3+2]*(p_indx==ReflecZ?-1.0:1.0))<eps){
        for(int k=0;k<dof;k++){
          P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
        }
      }
    }
  }else if(p_indx==SwapXY || p_indx==SwapXZ){
    for(int i=0;i<n_trg;i++)
    for(int j=0;j<n_trg;j++){
      if(pvfmm::fabs<Real_t>(trg_coord[i*3+0]-trg_coord[j*3+(p_indx==SwapXY?1:2)])<eps)
      if(pvfmm::fabs<Real_t>(trg_coord[i*3+1]-trg_coord[j*3+(p_indx==SwapXY?0:1)])<eps)
      if(pvfmm::fabs<Real_t>(trg_coord[i*3+2]-trg_coord[j*3+(p_indx==SwapXY?2:0)])<eps){
        for(int k=0;k<dof;k++){
          P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
        }
      }
    }
  }else{
    for(int j=0;j<n_trg;j++){
      for(int k=0;k<dof;k++){
        P.perm[j*dof+k]=j*dof+ker_perm.perm[k];
      }
    }
  }

  if(scal_exp && p_indx==Scaling){ // Set level-by-level scaling
    assert(dof==scal_exp->Dim());
    Vector<Real_t> scal(scal_exp->Dim());
    for(size_t i=0;i<scal.Dim();i++){
      scal[i]=pvfmm::pow<Real_t>(2.0,(*scal_exp)[i]);
    }
    for(int j=0;j<n_trg;j++){
      for(int i=0;i<dof;i++){
        P.scal[j*dof+i]*=scal[i];
      }
    }
  }
  { // Set P.scal
    for(int j=0;j<n_trg;j++){
      for(int i=0;i<dof;i++){
        P.scal[j*dof+i]*=ker_perm.scal[i];
      }
    }
  }

  return P;
}

template <class FMMNode>
Permutation<typename FMMNode::Real_t>& FMM_Pts<FMMNode>::PrecompPerm(Mat_Type type, Perm_Type perm_indx){

  //Check if the matrix already exists.
  Permutation<Real_t>& P_ = mat->Perm((Mat_Type)type, perm_indx);
  if(P_.Dim()!=0) return P_;


  size_t m=this->MultipoleOrder();
  size_t p_indx=perm_indx % C_Perm;

  //Compute the matrix.
  Permutation<Real_t> P;
  switch (type){
    case U2U_Type:
    {
      Vector<Real_t> scal_exp;
      Permutation<Real_t> ker_perm;
      if(perm_indx<C_Perm){ // Source permutation
        ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
        scal_exp=kernel->k_m2m->src_scal;
      }else{ // Target permutation
        ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
        scal_exp=kernel->k_m2m->src_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
      }
      P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      break;
    }
    case D2D_Type:
    {
      Vector<Real_t> scal_exp;
      Permutation<Real_t> ker_perm;
      if(perm_indx<C_Perm){ // Source permutation
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
      }else{ // Target permutation
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
      }
      P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      break;
    }
    default:
      break;
  }

  //Save the matrix for future use.
  #pragma omp critical (PRECOMP_MATRIX_PTS)
  {
    if(P_.Dim()==0) P_=P;
  }

  return P_;
}

template <class FMMNode>
Matrix<typename FMMNode::Real_t>& FMM_Pts<FMMNode>::Precomp(int level, Mat_Type type, size_t mat_indx){
  if(this->ScaleInvar()) level=0;

  //Check if the matrix already exists.
  Matrix<Real_t>& M_ = this->mat->Mat(level, type, mat_indx);
  if(M_.Dim(0)!=0 && M_.Dim(1)!=0) return M_;
  else{ //Compute matrix from symmetry class (if possible).
    size_t class_indx = this->interac_list.InteracClass(type, mat_indx);
    if(class_indx!=mat_indx){
      Matrix<Real_t>& M0 = this->Precomp(level, type, class_indx);
      if(M0.Dim(0)==0 || M0.Dim(1)==0) return M_;

      for(size_t i=0;i<Perm_Count;i++) this->PrecompPerm(type, (Perm_Type) i);
      Permutation<Real_t>& Pr = this->interac_list.Perm_R(abs(level), type, mat_indx);
      Permutation<Real_t>& Pc = this->interac_list.Perm_C(abs(level), type, mat_indx);
      if(Pr.Dim()>0 && Pc.Dim()>0 && M0.Dim(0)>0 && M0.Dim(1)>0) return M_;
    }
  }

  //Compute the matrix.
  Matrix<Real_t> M;
  //int omp_p=omp_get_max_threads();
  switch (type){

    case UC2UE0_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_m2m->ker_dim;
      // Coord of upward check surface
      Real_t c[3]={0,0,0};
      std::vector<Real_t> uc_coord=u_check_surf(MultipoleOrder(),c,level);
      size_t n_uc=uc_coord.size()/3;

      // Coord of upward equivalent surface
      std::vector<Real_t> ue_coord=u_equiv_surf(MultipoleOrder(),c,level);
      size_t n_ue=ue_coord.size()/3;

      // Evaluate potential at check surface due to equivalent surface.
      Matrix<Real_t> M_e2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
      kernel->k_m2m->BuildMatrix(&ue_coord[0], n_ue,
                             &uc_coord[0], n_uc, &(M_e2c[0][0]));

      Matrix<Real_t> U,S,V;
      M_e2c.SVD(U,S,V);
      Real_t eps=1, max_S=0;
      while(eps*(Real_t)0.5+(Real_t)1.0>1.0) eps*=0.5;
      for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
        if(pvfmm::fabs<Real_t>(S[i][i])>max_S) max_S=pvfmm::fabs<Real_t>(S[i][i]);
      }
      for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
      M=V.Transpose()*S;//*U.Transpose();
      break;
    }
    case UC2UE1_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_m2m->ker_dim;
      // Coord of upward check surface
      Real_t c[3]={0,0,0};
      std::vector<Real_t> uc_coord=u_check_surf(MultipoleOrder(),c,level);
      size_t n_uc=uc_coord.size()/3;

      // Coord of upward equivalent surface
      std::vector<Real_t> ue_coord=u_equiv_surf(MultipoleOrder(),c,level);
      size_t n_ue=ue_coord.size()/3;

      // Evaluate potential at check surface due to equivalent surface.
      Matrix<Real_t> M_e2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
      kernel->k_m2m->BuildMatrix(&ue_coord[0], n_ue,
                             &uc_coord[0], n_uc, &(M_e2c[0][0]));

      Matrix<Real_t> U,S,V;
      M_e2c.SVD(U,S,V);
      M=U.Transpose();
      break;
    }
    case DC2DE0_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_l2l->ker_dim;
      // Coord of downward check surface
      Real_t c[3]={0,0,0};
      std::vector<Real_t> check_surf=d_check_surf(MultipoleOrder(),c,level);
      size_t n_ch=check_surf.size()/3;

      // Coord of downward equivalent surface
      std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),c,level);
      size_t n_eq=equiv_surf.size()/3;

      // Evaluate potential at check surface due to equivalent surface.
      Matrix<Real_t> M_e2c(n_eq*ker_dim[0],n_ch*ker_dim[1]);
      kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_eq,
                             &check_surf[0], n_ch, &(M_e2c[0][0]));

      Matrix<Real_t> U,S,V;
      M_e2c.SVD(U,S,V);
      Real_t eps=1, max_S=0;
      while(eps*(Real_t)0.5+(Real_t)1.0>1.0) eps*=0.5;
      for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
        if(pvfmm::fabs<Real_t>(S[i][i])>max_S) max_S=pvfmm::fabs<Real_t>(S[i][i]);
      }
      for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
      M=V.Transpose()*S;//*U.Transpose();
      break;
    }
    case DC2DE1_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_l2l->ker_dim;
      // Coord of downward check surface
      Real_t c[3]={0,0,0};
      std::vector<Real_t> check_surf=d_check_surf(MultipoleOrder(),c,level);
      size_t n_ch=check_surf.size()/3;

      // Coord of downward equivalent surface
      std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),c,level);
      size_t n_eq=equiv_surf.size()/3;

      // Evaluate potential at check surface due to equivalent surface.
      Matrix<Real_t> M_e2c(n_eq*ker_dim[0],n_ch*ker_dim[1]);
      kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_eq,
                             &check_surf[0], n_ch, &(M_e2c[0][0]));

      Matrix<Real_t> U,S,V;
      M_e2c.SVD(U,S,V);
      M=U.Transpose();
      break;
    }
    case U2U_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_m2m->ker_dim;
      // Coord of upward check surface
      Real_t c[3]={0,0,0};
      std::vector<Real_t> check_surf=u_check_surf(MultipoleOrder(),c,level);
      size_t n_uc=check_surf.size()/3;

      // Coord of child's upward equivalent surface
      Real_t s=pvfmm::pow<Real_t>(0.5,(level+2));
      int* coord=interac_list.RelativeCoord(type,mat_indx);
      Real_t child_coord[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
      std::vector<Real_t> equiv_surf=u_equiv_surf(MultipoleOrder(),child_coord,level+1);
      size_t n_ue=equiv_surf.size()/3;

      // Evaluate potential at check surface due to equivalent surface.
      Matrix<Real_t> M_ce2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
      kernel->k_m2m->BuildMatrix(&equiv_surf[0], n_ue,
                             &check_surf[0], n_uc, &(M_ce2c[0][0]));
      Matrix<Real_t>& M_c2e0 = Precomp(level, UC2UE0_Type, 0);
      Matrix<Real_t>& M_c2e1 = Precomp(level, UC2UE1_Type, 0);
      M=(M_ce2c*M_c2e0)*M_c2e1;
      break;
    }
    case D2D_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_l2l->ker_dim;
      // Coord of downward check surface
      Real_t s=pvfmm::pow<Real_t>(0.5,level+1);
      int* coord=interac_list.RelativeCoord(type,mat_indx);
      Real_t c[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
      std::vector<Real_t> check_surf=d_check_surf(MultipoleOrder(),c,level);
      size_t n_dc=check_surf.size()/3;

      // Coord of parent's downward equivalent surface
      Real_t parent_coord[3]={0,0,0};
      std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),parent_coord,level-1);
      size_t n_de=equiv_surf.size()/3;

      // Evaluate potential at check surface due to equivalent surface.
      Matrix<Real_t> M_pe2c(n_de*ker_dim[0],n_dc*ker_dim[1]);
      kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_de,
                             &check_surf[0], n_dc, &(M_pe2c[0][0]));
      Matrix<Real_t> M_c2e0=Precomp(level-1,DC2DE0_Type,0);
      Matrix<Real_t> M_c2e1=Precomp(level-1,DC2DE1_Type,0);
      if(ScaleInvar()){ // Scale M_c2e0 for level-1
        Permutation<Real_t> ker_perm=this->kernel->k_l2l->perm_vec[C_Perm+Scaling];
        Vector<Real_t> scal_exp=this->kernel->k_l2l->trg_scal;
        Permutation<Real_t> P=equiv_surf_perm(MultipoleOrder(), Scaling, ker_perm, &scal_exp);
        M_c2e0=P*M_c2e0;
      }
      if(ScaleInvar()){ // Scale M_c2e1 for level-1
        Permutation<Real_t> ker_perm=this->kernel->k_l2l->perm_vec[0     +Scaling];
        Vector<Real_t> scal_exp=this->kernel->k_l2l->src_scal;
        Permutation<Real_t> P=equiv_surf_perm(MultipoleOrder(), Scaling, ker_perm, &scal_exp);
        M_c2e1=M_c2e1*P;
      }
      M=M_c2e0*(M_c2e1*M_pe2c);
      break;
    }
    case D2T_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_l2t->ker_dim;
      std::vector<Real_t>& rel_trg_coord=mat->RelativeTrgCoord();

      // Coord of target points
      Real_t r=pvfmm::pow<Real_t>(0.5,level);
      size_t n_trg=rel_trg_coord.size()/3;
      std::vector<Real_t> trg_coord(n_trg*3);
      for(size_t i=0;i<n_trg*COORD_DIM;i++) trg_coord[i]=rel_trg_coord[i]*r;

      // Coord of downward equivalent surface
      Real_t c[3]={0,0,0};
      std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),c,level);
      size_t n_eq=equiv_surf.size()/3;

      // Evaluate potential at target points due to equivalent surface.
      {
        M     .Resize(n_eq*ker_dim [0], n_trg*ker_dim [1]);
        kernel->k_l2t->BuildMatrix(&equiv_surf[0], n_eq, &trg_coord[0], n_trg, &(M     [0][0]));
      }
      Matrix<Real_t>& M_c2e0=Precomp(level,DC2DE0_Type,0);
      Matrix<Real_t>& M_c2e1=Precomp(level,DC2DE1_Type,0);
      M=M_c2e0*(M_c2e1*M);
      break;
    }
    case V_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_m2l->ker_dim;
      int n1=MultipoleOrder()*2;
      int n3 =n1*n1*n1;
      int n3_=n1*n1*(n1/2+1);

      //Compute the matrix.
      Real_t s=pvfmm::pow<Real_t>(0.5,level);
      int* coord2=interac_list.RelativeCoord(type,mat_indx);
      Real_t coord_diff[3]={coord2[0]*s,coord2[1]*s,coord2[2]*s};

      //Evaluate potential.
      std::vector<Real_t> r_trg(COORD_DIM,0.0);
      std::vector<Real_t> conv_poten(n3*ker_dim[0]*ker_dim[1]);
      std::vector<Real_t> conv_coord=conv_grid(MultipoleOrder(),coord_diff,level);
      kernel->k_m2l->BuildMatrix(&conv_coord[0],n3,&r_trg[0],1,&conv_poten[0]);

      //Rearrange data.
      Matrix<Real_t> M_conv(n3,ker_dim[0]*ker_dim[1],&conv_poten[0],false);
      M_conv=M_conv.Transpose();

      //Compute FFTW plan.
      int nnn[3]={n1,n1,n1};
      Real_t *fftw_in, *fftw_out;
      fftw_in  = mem::aligned_new<Real_t>(  n3 *ker_dim[0]*ker_dim[1]*sizeof(Real_t));
      fftw_out = mem::aligned_new<Real_t>(2*n3_*ker_dim[0]*ker_dim[1]*sizeof(Real_t));
      #pragma omp critical (FFTW_PLAN)
      {
        if (!vprecomp_fft_flag){
          vprecomp_fftplan = FFTW_t<Real_t>::fft_plan_many_dft_r2c(COORD_DIM, nnn, ker_dim[0]*ker_dim[1],
              (Real_t*)fftw_in, NULL, 1, n3, (typename FFTW_t<Real_t>::cplx*) fftw_out, NULL, 1, n3_);
          vprecomp_fft_flag=true;
        }
      }

      //Compute FFT.
      mem::memcopy(fftw_in, &conv_poten[0], n3*ker_dim[0]*ker_dim[1]*sizeof(Real_t));
      FFTW_t<Real_t>::fft_execute_dft_r2c(vprecomp_fftplan, (Real_t*)fftw_in, (typename FFTW_t<Real_t>::cplx*)(fftw_out));
      Matrix<Real_t> M_(2*n3_*ker_dim[0]*ker_dim[1],1,(Real_t*)fftw_out,false);
      M=M_;

      //Free memory.
      mem::aligned_delete<Real_t>(fftw_in);
      mem::aligned_delete<Real_t>(fftw_out);
      break;
    }
    case V1_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_m2l->ker_dim;
      size_t mat_cnt =interac_list.ListCount( V_Type);
      for(size_t k=0;k<mat_cnt;k++) Precomp(level, V_Type, k);

      const size_t chld_cnt=1UL<<COORD_DIM;
      size_t n1=MultipoleOrder()*2;
      size_t M_dim=n1*n1*(n1/2+1);
      size_t n3=n1*n1*n1;

      Vector<Real_t> zero_vec(M_dim*ker_dim[0]*ker_dim[1]*2);
      zero_vec.SetZero();

      Vector<Real_t*> M_ptr(chld_cnt*chld_cnt);
      for(size_t i=0;i<chld_cnt*chld_cnt;i++) M_ptr[i]=&zero_vec[0];

      int* rel_coord_=interac_list.RelativeCoord(V1_Type, mat_indx);
      for(int j1=0;j1<chld_cnt;j1++)
      for(int j2=0;j2<chld_cnt;j2++){
        int rel_coord[3]={rel_coord_[0]*2-(j1/1)%2+(j2/1)%2,
                          rel_coord_[1]*2-(j1/2)%2+(j2/2)%2,
                          rel_coord_[2]*2-(j1/4)%2+(j2/4)%2};
        for(size_t k=0;k<mat_cnt;k++){
          int* ref_coord=interac_list.RelativeCoord(V_Type, k);
          if(ref_coord[0]==rel_coord[0] &&
             ref_coord[1]==rel_coord[1] &&
             ref_coord[2]==rel_coord[2]){
            Matrix<Real_t>& M = this->mat->Mat(level, V_Type, k);
            M_ptr[j2*chld_cnt+j1]=&M[0][0];
            break;
          }
        }
      }

      // Build matrix ker_dim0 x ker_dim1 x M_dim x 8 x 8
      M.Resize(ker_dim[0]*ker_dim[1]*M_dim, 2*chld_cnt*chld_cnt);
      for(int j=0;j<ker_dim[0]*ker_dim[1]*M_dim;j++){
        for(size_t k=0;k<chld_cnt*chld_cnt;k++){
          M[j][k*2+0]=M_ptr[k][j*2+0]/n3;
          M[j][k*2+1]=M_ptr[k][j*2+1]/n3;
        }
      }
      break;
    }
    case W_Type:
    {
      if(MultipoleOrder()==0) break;
      const int* ker_dim=kernel->k_m2t->ker_dim;
      std::vector<Real_t>& rel_trg_coord=mat->RelativeTrgCoord();

      // Coord of target points
      Real_t s=pvfmm::pow<Real_t>(0.5,level);
      size_t n_trg=rel_trg_coord.size()/3;
      std::vector<Real_t> trg_coord(n_trg*3);
      for(size_t j=0;j<n_trg*COORD_DIM;j++) trg_coord[j]=rel_trg_coord[j]*s;

      // Coord of downward equivalent surface
      int* coord2=interac_list.RelativeCoord(type,mat_indx);
      Real_t c[3]={(Real_t)((coord2[0]+1)*s*0.25),(Real_t)((coord2[1]+1)*s*0.25),(Real_t)((coord2[2]+1)*s*0.25)};
      std::vector<Real_t> equiv_surf=u_equiv_surf(MultipoleOrder(),c,level+1);
      size_t n_eq=equiv_surf.size()/3;

      // Evaluate potential at target points due to equivalent surface.
      {
        M     .Resize(n_eq*ker_dim [0],n_trg*ker_dim [1]);
        kernel->k_m2t->BuildMatrix(&equiv_surf[0], n_eq, &trg_coord[0], n_trg, &(M     [0][0]));
      }
      break;
    }
    case BC_Type:
    {
      if(!this->ScaleInvar() || MultipoleOrder()==0) break;
      if(kernel->k_m2l->ker_dim[0]!=kernel->k_m2m->ker_dim[0]) break;
      if(kernel->k_m2l->ker_dim[1]!=kernel->k_l2l->ker_dim[1]) break;
      int ker_dim[2]={kernel->k_m2l->ker_dim[0],kernel->k_m2l->ker_dim[1]};
      size_t n_surf=(6*(MultipoleOrder()-1)*(MultipoleOrder()-1)+2);  //Total number of points.

      if((M.Dim(0)!=n_surf*ker_dim[0] || M.Dim(1)!=n_surf*ker_dim[1]) && level==0){
        { // Compute M
          M.ReInit(n_surf*ker_dim[0], n_surf*ker_dim[1]); M.SetZero();

          Real_t dc_coord[3]={0,0,0};
          std::vector<Real_t> trg_coord=d_check_surf(MultipoleOrder(), dc_coord, 0);

          int xlow,xhigh,ylow,yhigh,zlow,zhigh;
          switch(periodicType){
          case PeriodicType::NONE :
        	  xlow=0;xhigh=0;ylow=0;yhigh=0;zlow=0;zhigh=0;
        	  break;
          case PeriodicType::PXYZ :
        	  xlow=-BC_LEVELS-1;xhigh=BC_LEVELS+1;
        	  ylow=-BC_LEVELS-1;yhigh=BC_LEVELS+1;
        	  zlow=-BC_LEVELS-1;zhigh=BC_LEVELS+1;
        	  break;
          case PeriodicType::PZ :
        	  xlow=0;xhigh=0;
        	  ylow=0;yhigh=0;
        	  zlow=-BC_LEVELS-1;zhigh=BC_LEVELS+1;
        	  break;
          case PeriodicType::PXY :
        	  xlow=-BC_LEVELS-1;xhigh=BC_LEVELS+1;
        	  ylow=-BC_LEVELS-1;yhigh=BC_LEVELS+1;
        	  zlow=0;zhigh=0;
        	  break;
          }

          for(int x0=xlow;x0<=xhigh;x0++)
          for(int x1=ylow;x1<=yhigh;x1++)
          for(int x2=zlow;x2<=zhigh;x2++)
          if(abs(x0)>1 || abs(x1)>1 || abs(x2)>1){
            Real_t ue_coord[3]={(Real_t)x0, (Real_t)x1, (Real_t)x2};
            std::vector<Real_t> src_coord=u_equiv_surf(MultipoleOrder(), ue_coord, 0);

            Matrix<Real_t> M_tmp(n_surf*ker_dim[0], n_surf*ker_dim[1]);
            kernel->k_m2l->BuildMatrix(&src_coord[0], n_surf,
                                       &trg_coord[0], n_surf, &(M_tmp[0][0]));
            M+=M_tmp;
          }
        }

        if(!this->ScaleInvar()){ // Free memory
          Mat_Type type=D2D_Type;
          for(int l=-BC_LEVELS;l<0;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=U2U_Type;
          for(int l=-BC_LEVELS;l<0;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=DC2DE0_Type;
          for(int l=-BC_LEVELS;l<0;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=DC2DE1_Type;
          for(int l=-BC_LEVELS;l<0;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=UC2UE0_Type;
          for(int l=-BC_LEVELS;l<0;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=UC2UE1_Type;
          for(int l=-BC_LEVELS;l<0;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
        }
      }

      break;
    }
    default:
      break;
  }

  //Save the matrix for future use.
  #pragma omp critical (PRECOMP_MATRIX_PTS)
  if(M_.Dim(0)==0 && M_.Dim(1)==0){
    M_=M;
    /*
    M_.Resize(M.Dim(0),M.Dim(1));
    int dof=ker_dim[0]*ker_dim[1];
    for(int j=0;j<dof;j++){
      size_t a=(M.Dim(0)*M.Dim(1)* j   )/dof;
      size_t b=(M.Dim(0)*M.Dim(1)*(j+1))/dof;
      #pragma omp parallel for // NUMA
      for(int tid=0;tid<omp_p;tid++){
        size_t a_=a+((b-a)* tid   )/omp_p;
        size_t b_=a+((b-a)*(tid+1))/omp_p;
        mem::memcopy(&M_[0][a_], &M[0][a_], (b_-a_)*sizeof(Real_t));
      }
    }
    */
  }

  return M_;
}

template <class FMMNode>
void FMM_Pts<FMMNode>::PrecompAll(Mat_Type type, int level){
  if(level==-1){
    for(int l=0;l<MAX_DEPTH;l++){
      PrecompAll(type, l);
    }
    return;
  }

  //Compute basic permutations.
  for(size_t i=0;i<Perm_Count;i++)
    this->PrecompPerm(type, (Perm_Type) i);

  {
    //Allocate matrices.
    size_t mat_cnt=interac_list.ListCount((Mat_Type)type);
    mat->Mat(level, (Mat_Type)type, mat_cnt-1);

    { // Compute InteracClass matrices.
      std::vector<size_t> indx_lst;
      for(size_t i=0; i<mat_cnt; i++){
        if(interac_list.InteracClass((Mat_Type)type,i)==i)
          indx_lst.push_back(i);
      }

      //Compute Transformations.
      //#pragma omp parallel for //lets use fine grained parallelism
      for(size_t i=0; i<indx_lst.size(); i++){
        Precomp(level, (Mat_Type)type, indx_lst[i]);
      }
    }

    //#pragma omp parallel for //lets use fine grained parallelism
    for(size_t mat_indx=0;mat_indx<mat_cnt;mat_indx++){
      Matrix<Real_t>& M0=interac_list.ClassMat(level,(Mat_Type)type,mat_indx);
      Permutation<Real_t>& pr=interac_list.Perm_R(abs(level), (Mat_Type)type, mat_indx);
      Permutation<Real_t>& pc=interac_list.Perm_C(abs(level), (Mat_Type)type, mat_indx);
      if(pr.Dim()!=M0.Dim(0) || pc.Dim()!=M0.Dim(1)) Precomp(level, (Mat_Type)type, mat_indx);
    }
  }
}

template <class FMMNode>
void FMM_Pts<FMMNode>::CollectNodeData(FMMTree_t* tree, std::vector<FMMNode*>& node, std::vector<Matrix<Real_t> >& buff_list, std::vector<Vector<FMMNode_t*> >& n_list, std::vector<std::vector<Vector<Real_t>* > > vec_list){
  if(buff_list.size()<7) buff_list.resize(7);
  if(   n_list.size()<7)    n_list.resize(7);
  if( vec_list.size()<7)  vec_list.resize(7);
  int omp_p=omp_get_max_threads();

  if(node.size()==0) return;
  {// 0. upward_equiv
    int indx=0;

    size_t vec_sz;
    { // Set vec_sz
      Matrix<Real_t>& M_uc2ue = this->interac_list.ClassMat(0, UC2UE1_Type, 0);
      vec_sz=M_uc2ue.Dim(1);
    }

    std::vector< FMMNode* > node_lst;
    {// Construct node_lst
      node_lst.clear();
      std::vector<std::vector< FMMNode* > > node_lst_(MAX_DEPTH+1);
      FMMNode_t* r_node=NULL;
      for(size_t i=0;i<node.size();i++){
        if(!node[i]->IsLeaf()){
          node_lst_[node[i]->Depth()].push_back(node[i]);
        }else{
          node[i]->pt_cnt[0]+=node[i]-> src_coord.Dim()/COORD_DIM;
          node[i]->pt_cnt[0]+=node[i]->surf_coord.Dim()/COORD_DIM;
          if(node[i]->IsGhost()) node[i]->pt_cnt[0]++; // TODO: temporary fix, pt_cnt not known for ghost nodes
        }
        if(node[i]->Depth()==0) r_node=node[i];
      }
      size_t chld_cnt=1UL<<COORD_DIM;
      for(int i=MAX_DEPTH;i>=0;i--){
        for(size_t j=0;j<node_lst_[i].size();j++){
          for(size_t k=0;k<chld_cnt;k++){
            FMMNode_t* node=(FMMNode_t*)node_lst_[i][j]->Child(k);
            node_lst_[i][j]->pt_cnt[0]+=node->pt_cnt[0];
          }
        }
      }
      for(int i=0;i<=MAX_DEPTH;i++){
        for(size_t j=0;j<node_lst_[i].size();j++){
          if(node_lst_[i][j]->pt_cnt[0]){
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=(FMMNode_t*)node_lst_[i][j]->Child(k);
              node_lst.push_back(node);
            }
          }else{
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=(FMMNode_t*)node_lst_[i][j]->Child(k);
              node->FMMData()->upward_equiv.ReInit(0);
            }
          }
        }
      }
      if(r_node!=NULL) node_lst.push_back(r_node);
      n_list[indx]=node_lst;
    }

    std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){ // Construct vec_lst
      FMMNode_t* node=node_lst[i];
      Vector<Real_t>& data_vec=node->FMMData()->upward_equiv;
      data_vec.ReInit(vec_sz,NULL,false);
      vec_lst.push_back(&data_vec);
    }
  }
  {// 1. dnward_equiv
    int indx=1;

    size_t vec_sz;
    { // Set vec_sz
      Matrix<Real_t>& M_dc2de0 = this->interac_list.ClassMat(0, DC2DE0_Type, 0);
      vec_sz=M_dc2de0.Dim(0);
    }

    std::vector< FMMNode* > node_lst;
    {// Construct node_lst
      node_lst.clear();
      std::vector<std::vector< FMMNode* > > node_lst_(MAX_DEPTH+1);
      FMMNode_t* r_node=NULL;
      for(size_t i=0;i<node.size();i++){
        if(!node[i]->IsLeaf()){
          node_lst_[node[i]->Depth()].push_back(node[i]);
        }else{
          node[i]->pt_cnt[1]+=node[i]->trg_coord.Dim()/COORD_DIM;
        }
        if(node[i]->Depth()==0) r_node=node[i];
      }
      size_t chld_cnt=1UL<<COORD_DIM;
      for(int i=MAX_DEPTH;i>=0;i--){
        for(size_t j=0;j<node_lst_[i].size();j++){
          for(size_t k=0;k<chld_cnt;k++){
            FMMNode_t* node=(FMMNode_t*)node_lst_[i][j]->Child(k);
            node_lst_[i][j]->pt_cnt[1]+=node->pt_cnt[1];
          }
        }
      }
      for(int i=0;i<=MAX_DEPTH;i++){
        for(size_t j=0;j<node_lst_[i].size();j++){
          if(node_lst_[i][j]->pt_cnt[1]){
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=(FMMNode_t*)node_lst_[i][j]->Child(k);
              node_lst.push_back(node);
            }
          }else{
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=(FMMNode_t*)node_lst_[i][j]->Child(k);
              node->FMMData()->dnward_equiv.ReInit(0);
            }
          }
        }
      }
      if(r_node!=NULL) node_lst.push_back(r_node);
      n_list[indx]=node_lst;
    }

    std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){ // Construct vec_lst
      FMMNode_t* node=node_lst[i];
      Vector<Real_t>& data_vec=node->FMMData()->dnward_equiv;
      data_vec.ReInit(vec_sz,NULL,false);
      vec_lst.push_back(&data_vec);
    }
  }
  {// 2. upward_equiv_fft
    int indx=2;
    std::vector< FMMNode* > node_lst;
    {
      std::vector<std::vector< FMMNode* > > node_lst_(MAX_DEPTH+1);
      for(size_t i=0;i<node.size();i++)
        if(!node[i]->IsLeaf())
          node_lst_[node[i]->Depth()].push_back(node[i]);
      for(int i=0;i<=MAX_DEPTH;i++)
        for(size_t j=0;j<node_lst_[i].size();j++)
          node_lst.push_back(node_lst_[i][j]);
    }
    n_list[indx]=node_lst;
  }
  {// 3. dnward_check_fft
    int indx=3;
    std::vector< FMMNode* > node_lst;
    {
      std::vector<std::vector< FMMNode* > > node_lst_(MAX_DEPTH+1);
      for(size_t i=0;i<node.size();i++)
        if(!node[i]->IsLeaf() && !node[i]->IsGhost())
          node_lst_[node[i]->Depth()].push_back(node[i]);
      for(int i=0;i<=MAX_DEPTH;i++)
        for(size_t j=0;j<node_lst_[i].size();j++)
          node_lst.push_back(node_lst_[i][j]);
    }
    n_list[indx]=node_lst;
  }
  {// 4. src_val
    int indx=4;
    int src_dof=kernel->ker_dim[0];
    int surf_dof=COORD_DIM+src_dof;

    std::vector< FMMNode* > node_lst;
    for(size_t i=0;i<node.size();i++){// Construct node_lst
      if(node[i]->IsLeaf()){
        node_lst.push_back(node[i]);
      }else{
        node[i]->src_value.ReInit(0);
        node[i]->surf_value.ReInit(0);
      }
    }
    n_list[indx]=node_lst;

    std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){ // Construct vec_lst
      FMMNode_t* node=node_lst[i];
      { // src_value
        Vector<Real_t>& data_vec=node->src_value;
        size_t vec_sz=(node->src_coord.Dim()/COORD_DIM)*src_dof;
        if(data_vec.Dim()!=vec_sz) data_vec.ReInit(vec_sz,NULL,false);
        vec_lst.push_back(&data_vec);
      }
      { // surf_value
        Vector<Real_t>& data_vec=node->surf_value;
        size_t vec_sz=(node->surf_coord.Dim()/COORD_DIM)*surf_dof;
        if(data_vec.Dim()!=vec_sz) data_vec.ReInit(vec_sz,NULL,false);
        vec_lst.push_back(&data_vec);
      }
    }
  }
  {// 5. trg_val
    int indx=5;
    int trg_dof=kernel->ker_dim[1];

    std::vector< FMMNode* > node_lst;
    for(size_t i=0;i<node.size();i++){// Construct node_lst
      if(node[i]->IsLeaf() && !node[i]->IsGhost()){
        node_lst.push_back(node[i]);
      }else{
        node[i]->trg_value.ReInit(0);
      }
    }
    n_list[indx]=node_lst;

    std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){ // Construct vec_lst
      FMMNode_t* node=node_lst[i];
      { // trg_value
        Vector<Real_t>& data_vec=node->trg_value;
        size_t vec_sz=(node->trg_coord.Dim()/COORD_DIM)*trg_dof;
        data_vec.ReInit(vec_sz,NULL,false);
        vec_lst.push_back(&data_vec);
      }
    }
  }
  {// 6. pts_coord
    int indx=6;

    std::vector< FMMNode* > node_lst;
    for(size_t i=0;i<node.size();i++){// Construct node_lst
      if(node[i]->IsLeaf()){
        node_lst.push_back(node[i]);
      }else{
        node[i]->src_coord.ReInit(0);
        node[i]->surf_coord.ReInit(0);
        node[i]->trg_coord.ReInit(0);
      }
    }
    n_list[indx]=node_lst;

    std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){ // Construct vec_lst
      FMMNode_t* node=node_lst[i];
      { // src_coord
        Vector<Real_t>& data_vec=node->src_coord;
        vec_lst.push_back(&data_vec);
      }
      { // surf_coord
        Vector<Real_t>& data_vec=node->surf_coord;
        vec_lst.push_back(&data_vec);
      }
      { // trg_coord
        Vector<Real_t>& data_vec=node->trg_coord;
        vec_lst.push_back(&data_vec);
      }
    }
    { // check and equiv surfaces.
      if(tree->upwd_check_surf.size()==0){
        size_t m=MultipoleOrder();
        tree->upwd_check_surf.resize(MAX_DEPTH);
        tree->upwd_equiv_surf.resize(MAX_DEPTH);
        tree->dnwd_check_surf.resize(MAX_DEPTH);
        tree->dnwd_equiv_surf.resize(MAX_DEPTH);
        for(size_t depth=0;depth<MAX_DEPTH;depth++){
          Real_t c[3]={0.0,0.0,0.0};
          tree->upwd_check_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
          tree->upwd_equiv_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
          tree->dnwd_check_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
          tree->dnwd_equiv_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
          tree->upwd_check_surf[depth]=u_check_surf(m,c,depth);
          tree->upwd_equiv_surf[depth]=u_equiv_surf(m,c,depth);
          tree->dnwd_check_surf[depth]=d_check_surf(m,c,depth);
          tree->dnwd_equiv_surf[depth]=d_equiv_surf(m,c,depth);
        }
      }
      for(size_t depth=0;depth<MAX_DEPTH;depth++){
        vec_lst.push_back(&tree->upwd_check_surf[depth]);
        vec_lst.push_back(&tree->upwd_equiv_surf[depth]);
        vec_lst.push_back(&tree->dnwd_check_surf[depth]);
        vec_lst.push_back(&tree->dnwd_equiv_surf[depth]);
      }
    }
  }

  // Create extra auxiliary buffer.
  if(buff_list.size()<=vec_list.size()) buff_list.resize(vec_list.size()+1);
  for(size_t indx=0;indx<vec_list.size();indx++){ // Resize buffer
    Matrix<Real_t>&                  buff=buff_list[indx];
    std::vector<Vector<Real_t>*>& vec_lst= vec_list[indx];
    bool keep_data=(indx==4 || indx==6);
    size_t n_vec=vec_lst.size();

    { // Continue if nothing to be done.
      if(!n_vec) continue;
      if(buff.Dim(0)*buff.Dim(1)>0){
        bool init_buff=false;
        Real_t* buff_start=buff.Begin();
        Real_t* buff_end=buff.Begin()+buff.Dim(0)*buff.Dim(1);
        #pragma omp parallel for reduction(||:init_buff)
        for(size_t i=0;i<n_vec;i++){
          if(vec_lst[i]->Dim() && (vec_lst[i]->Begin()<buff_start || vec_lst[i]->Begin()>=buff_end)){
            init_buff=true;
          }
        }
        if(!init_buff) continue;
      }
    }

    std::vector<size_t> vec_size(n_vec);
    std::vector<size_t> vec_disp(n_vec);
    if(n_vec){ // Set vec_size and vec_disp
      #pragma omp parallel for
      for(size_t i=0;i<n_vec;i++){ // Set vec_size
        vec_size[i]=vec_lst[i]->Dim();
      }

      vec_disp[0]=0;
      omp_par::scan(&vec_size[0],&vec_disp[0],n_vec);
    }
    size_t buff_size=vec_size[n_vec-1]+vec_disp[n_vec-1];
    if(!buff_size) continue;

    if(keep_data){ // Copy to dev_buffer
      if(dev_buffer.Dim()<buff_size*sizeof(Real_t)){ // Resize dev_buffer
        dev_buffer.ReInit(buff_size*sizeof(Real_t)*1.05);
      }

      #pragma omp parallel for
      for(size_t i=0;i<n_vec;i++){
        if(vec_lst[i]->Begin()){
          mem::memcopy(((Real_t*)dev_buffer.Begin())+vec_disp[i],vec_lst[i]->Begin(),vec_size[i]*sizeof(Real_t));
        }
      }
    }

    if(buff.Dim(0)*buff.Dim(1)<buff_size){ // Resize buff
      buff.ReInit(1,buff_size*1.05);
    }

    if(keep_data){ // Copy to buff (from dev_buffer)
      #pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        size_t a=(buff_size*(tid+0))/omp_p;
        size_t b=(buff_size*(tid+1))/omp_p;
        mem::memcopy(buff.Begin()+a,((Real_t*)dev_buffer.Begin())+a,(b-a)*sizeof(Real_t));
      }
    }

    #pragma omp parallel for
    for(size_t i=0;i<n_vec;i++){ // ReInit vectors
      vec_lst[i]->ReInit(vec_size[i],buff.Begin()+vec_disp[i],false);
    }
  }
}



template <class FMMNode>
void FMM_Pts<FMMNode>::SetupPrecomp(SetupData<Real_t>& setup_data, bool device){
  if(setup_data.precomp_data==NULL || setup_data.level>MAX_DEPTH) return;

  Profile::Tic("SetupPrecomp",&this->comm,true,25);
  { // Build precomp_data
    size_t precomp_offset=0;
    int level=setup_data.level;
    Matrix<char>& precomp_data=*setup_data.precomp_data;
    std::vector<Mat_Type>& interac_type_lst=setup_data.interac_type;
    for(size_t type_indx=0; type_indx<interac_type_lst.size(); type_indx++){
      Mat_Type& interac_type=interac_type_lst[type_indx];
      this->PrecompAll(interac_type, level); // Compute matrices.
      precomp_offset=this->mat->CompactData(level, interac_type, precomp_data, precomp_offset);
    }
  }
  Profile::Toc();

  if(device){ // Host2Device
    Profile::Tic("Host2Device",&this->comm,false,25);
    setup_data.precomp_data->AllocDevice(true);
    Profile::Toc();
  }
}

template <class FMMNode>
void FMM_Pts<FMMNode>::SetupInterac(SetupData<Real_t>& setup_data, bool device){
  int level=setup_data.level;
  std::vector<Mat_Type>& interac_type_lst=setup_data.interac_type;

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  Matrix<Real_t>&  input_data=*setup_data. input_data;
  Matrix<Real_t>& output_data=*setup_data.output_data;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector;

  size_t n_in =nodes_in .size();
  size_t n_out=nodes_out.size();

  // Setup precomputed data.
  if(setup_data.precomp_data->Dim(0)*setup_data.precomp_data->Dim(1)==0) SetupPrecomp(setup_data,device);

  // Build interac_data
  Profile::Tic("Interac-Data",&this->comm,true,25);
  Matrix<char>& interac_data=setup_data.interac_data;
  { // Build precomp_data, interac_data
    std::vector<size_t> interac_mat;
    std::vector<size_t> interac_cnt;
    std::vector<size_t> interac_blk;
    std::vector<size_t>  input_perm;
    std::vector<size_t> output_perm;
    size_t dof=0, M_dim0=0, M_dim1=0;

    size_t precomp_offset=0;
    size_t buff_size=DEVICE_BUFFER_SIZE*1024l*1024l;
    if(n_out && n_in) for(size_t type_indx=0; type_indx<interac_type_lst.size(); type_indx++){

      Mat_Type& interac_type=interac_type_lst[type_indx];
      size_t mat_cnt=this->interac_list.ListCount(interac_type);
      Matrix<size_t> precomp_data_offset;
      { // Load precomp_data for interac_type.
        struct HeaderData{
          size_t total_size;
          size_t      level;
          size_t   mat_cnt ;
          size_t  max_depth;
        };
        Matrix<char>& precomp_data=*setup_data.precomp_data;
        char* indx_ptr=precomp_data[0]+precomp_offset;
        HeaderData& header=*(HeaderData*)indx_ptr;indx_ptr+=sizeof(HeaderData);
        precomp_data_offset.ReInit(header.mat_cnt,(1+(2+2)*header.max_depth), (size_t*)indx_ptr, false);
        precomp_offset+=header.total_size;
      }

      Matrix<FMMNode*> src_interac_list(n_in ,mat_cnt); src_interac_list.SetZero();
      Matrix<FMMNode*> trg_interac_list(n_out,mat_cnt); trg_interac_list.SetZero();
      { // Build trg_interac_list
        #pragma omp parallel for
        for(size_t i=0;i<n_out;i++){
          if(!((FMMNode*)nodes_out[i])->IsGhost() && (level==-1 || ((FMMNode*)nodes_out[i])->Depth()==level)){
            Vector<FMMNode*>& lst=((FMMNode*)nodes_out[i])->interac_list[interac_type];
            mem::memcopy(trg_interac_list[i], lst.Begin(), lst.Dim()*sizeof(FMMNode*));
            assert(lst.Dim()==mat_cnt);
          }
        }
      }
      { // Build src_interac_list
        #pragma omp parallel for
        for(size_t i=0;i<n_out;i++){
          for(size_t j=0;j<mat_cnt;j++)
          if(trg_interac_list[i][j]!=NULL){
            trg_interac_list[i][j]->node_id=n_in;
          }
        }
        #pragma omp parallel for
        for(size_t i=0;i<n_in ;i++) ((FMMNode*)nodes_in [i])->node_id=i;
        #pragma omp parallel for
        for(size_t i=0;i<n_out;i++){
          for(size_t j=0;j<mat_cnt;j++){
            if(trg_interac_list[i][j]!=NULL){
              if(trg_interac_list[i][j]->node_id==n_in){
                trg_interac_list[i][j]=NULL;
              }else{
                src_interac_list[trg_interac_list[i][j]->node_id][j]=(FMMNode*)nodes_out[i];
              }
            }
          }
        }
      }

      Matrix<size_t> interac_dsp(n_out,mat_cnt);
      std::vector<size_t> interac_blk_dsp(1,0);
      { // Determine dof, M_dim0, M_dim1
        dof=1;
        Matrix<Real_t>& M0 = this->interac_list.ClassMat(level, interac_type_lst[0], 0);
        M_dim0=M0.Dim(0); M_dim1=M0.Dim(1);
      }
      { // Determine interaction blocks which fit in memory.
        size_t vec_size=(M_dim0+M_dim1)*sizeof(Real_t)*dof;
        for(size_t j=0;j<mat_cnt;j++){// Determine minimum buff_size
          size_t vec_cnt=0;
          for(size_t i=0;i<n_out;i++){
            if(trg_interac_list[i][j]!=NULL) vec_cnt++;
          }
          if(buff_size<vec_cnt*vec_size)
            buff_size=vec_cnt*vec_size;
        }

        size_t interac_dsp_=0;
        for(size_t j=0;j<mat_cnt;j++){
          for(size_t i=0;i<n_out;i++){
            interac_dsp[i][j]=interac_dsp_;
            if(trg_interac_list[i][j]!=NULL) interac_dsp_++;
          }
          if(interac_dsp_*vec_size>buff_size) // Comment to disable symmetries.
          {
            interac_blk.push_back(j-interac_blk_dsp.back());
            interac_blk_dsp.push_back(j);

            size_t offset=interac_dsp[0][j];
            for(size_t i=0;i<n_out;i++) interac_dsp[i][j]-=offset;
            interac_dsp_-=offset;

            assert(interac_dsp_*vec_size<=buff_size); // Problem too big for buff_size.
          }
          interac_mat.push_back(precomp_data_offset[this->interac_list.InteracClass(interac_type,j)][0]);
          interac_cnt.push_back(interac_dsp_-interac_dsp[0][j]);
        }
        interac_blk.push_back(mat_cnt-interac_blk_dsp.back());
        interac_blk_dsp.push_back(mat_cnt);
      }

      { // Determine input_perm.
        size_t vec_size=M_dim0*dof;
        for(size_t i=0;i<n_out;i++) ((FMMNode*)nodes_out[i])->node_id=i;
        for(size_t k=1;k<interac_blk_dsp.size();k++){
          for(size_t i=0;i<n_in ;i++){
            for(size_t j=interac_blk_dsp[k-1];j<interac_blk_dsp[k];j++){
              FMMNode_t* trg_node=src_interac_list[i][j];
              if(trg_node!=NULL && trg_node->node_id<n_out){
                size_t depth=(this->ScaleInvar()?trg_node->Depth():0);
                input_perm .push_back(precomp_data_offset[j][1+4*depth+0]); // prem
                input_perm .push_back(precomp_data_offset[j][1+4*depth+1]); // scal
                input_perm .push_back(interac_dsp[trg_node->node_id][j]*vec_size*sizeof(Real_t)); // trg_ptr
                input_perm .push_back((size_t)(input_vector[i]->Begin()- input_data[0])); // src_ptr
                assert(input_vector[i]->Dim()==vec_size);
              }
            }
          }
        }
      }
      { // Determine output_perm
        size_t vec_size=M_dim1*dof;
        for(size_t k=1;k<interac_blk_dsp.size();k++){
          for(size_t i=0;i<n_out;i++){
            for(size_t j=interac_blk_dsp[k-1];j<interac_blk_dsp[k];j++){
              if(trg_interac_list[i][j]!=NULL){
                size_t depth=(this->ScaleInvar()?((FMMNode*)nodes_out[i])->Depth():0);
                output_perm.push_back(precomp_data_offset[j][1+4*depth+2]); // prem
                output_perm.push_back(precomp_data_offset[j][1+4*depth+3]); // scal
                output_perm.push_back(interac_dsp[               i ][j]*vec_size*sizeof(Real_t)); // src_ptr
                output_perm.push_back((size_t)(output_vector[i]->Begin()-output_data[0])); // trg_ptr
                assert(output_vector[i]->Dim()==vec_size);
              }
            }
          }
        }
      }
    }
    if(this->dev_buffer.Dim()<buff_size) this->dev_buffer.ReInit(buff_size);

    { // Set interac_data.
      size_t data_size=sizeof(size_t)*4;
      data_size+=sizeof(size_t)+interac_blk.size()*sizeof(size_t);
      data_size+=sizeof(size_t)+interac_cnt.size()*sizeof(size_t);
      data_size+=sizeof(size_t)+interac_mat.size()*sizeof(size_t);
      data_size+=sizeof(size_t)+ input_perm.size()*sizeof(size_t);
      data_size+=sizeof(size_t)+output_perm.size()*sizeof(size_t);
      if(interac_data.Dim(0)*interac_data.Dim(1)<sizeof(size_t)){
        data_size+=sizeof(size_t);
        interac_data.ReInit(1,data_size);
        ((size_t*)interac_data.Begin())[0]=sizeof(size_t);
      }else{
        size_t pts_data_size=*((size_t*)interac_data.Begin());
        assert(interac_data.Dim(0)*interac_data.Dim(1)>=pts_data_size);
        data_size+=pts_data_size;
        if(data_size>interac_data.Dim(0)*interac_data.Dim(1)){ //Resize and copy interac_data.
          Matrix< char> pts_interac_data=interac_data;
          interac_data.ReInit(1,data_size);
          mem::memcopy(interac_data.Begin(),pts_interac_data.Begin(),pts_data_size);
        }
      }
      char* data_ptr=interac_data.Begin();
      data_ptr+=((size_t*)data_ptr)[0];

      ((size_t*)data_ptr)[0]=data_size; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=   M_dim0; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=   M_dim1; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=      dof; data_ptr+=sizeof(size_t);

      ((size_t*)data_ptr)[0]=interac_blk.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &interac_blk[0], interac_blk.size()*sizeof(size_t));
      data_ptr+=interac_blk.size()*sizeof(size_t);

      ((size_t*)data_ptr)[0]=interac_cnt.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &interac_cnt[0], interac_cnt.size()*sizeof(size_t));
      data_ptr+=interac_cnt.size()*sizeof(size_t);

      ((size_t*)data_ptr)[0]=interac_mat.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &interac_mat[0], interac_mat.size()*sizeof(size_t));
      data_ptr+=interac_mat.size()*sizeof(size_t);

      ((size_t*)data_ptr)[0]= input_perm.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr,  &input_perm[0],  input_perm.size()*sizeof(size_t));
      data_ptr+= input_perm.size()*sizeof(size_t);

      ((size_t*)data_ptr)[0]=output_perm.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &output_perm[0], output_perm.size()*sizeof(size_t));
      data_ptr+=output_perm.size()*sizeof(size_t);
    }
  }
  Profile::Toc();

  if(device){ // Host2Device
    Profile::Tic("Host2Device",&this->comm,false,25);
    setup_data.interac_data .AllocDevice(true);
    if(staging_buffer.Dim()<sizeof(Real_t)*output_data.Dim(0)*output_data.Dim(1)){
      staging_buffer.ReInit(sizeof(Real_t)*output_data.Dim(0)*output_data.Dim(1));
      staging_buffer.SetZero();
      staging_buffer.AllocDevice(true);
    }
    Profile::Toc();
  }
}

#if defined(PVFMM_HAVE_CUDA)
#include <fmm_pts_gpu.hpp>

template <class Real_t, int SYNC>
void EvalListGPU(SetupData<Real_t>& setup_data, Vector<char>& dev_buffer, MPI_Comm& comm) {
  cudaStream_t* stream = pvfmm::CUDA_Lock::acquire_stream();

  Profile::Tic("Host2Device",&comm,false,25);
  typename Matrix<char>::Device    interac_data;
  typename Vector<char>::Device            buff;
  typename Matrix<char>::Device  precomp_data_d;
  typename Matrix<char>::Device  interac_data_d;
  typename Matrix<Real_t>::Device  input_data_d;
  typename Matrix<Real_t>::Device output_data_d;

  interac_data  = setup_data.interac_data;
  buff          =              dev_buffer. AllocDevice(false);
  precomp_data_d= setup_data.precomp_data->AllocDevice(false);
  interac_data_d= setup_data.interac_data. AllocDevice(false);
  input_data_d  = setup_data.  input_data->AllocDevice(false);
  output_data_d = setup_data. output_data->AllocDevice(false);
  Profile::Toc();

  Profile::Tic("DeviceComp",&comm,false,20);
  { // Offloaded computation.
    size_t data_size, M_dim0, M_dim1, dof;
    Vector<size_t> interac_blk;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_mat;
    Vector<size_t> input_perm_d;
    Vector<size_t> output_perm_d;

    { // Set interac_data.
      char* data_ptr=&interac_data  [0][0];
      char*  dev_ptr=&interac_data_d[0][0];

      data_size=((size_t*)data_ptr)[0]; data_ptr+=data_size;      dev_ptr += data_size;
      data_size=((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t); dev_ptr += sizeof(size_t);
      M_dim0   =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t); dev_ptr += sizeof(size_t);
      M_dim1   =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t); dev_ptr += sizeof(size_t);
      dof      =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t); dev_ptr += sizeof(size_t);

      interac_blk.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr += sizeof(size_t) + sizeof(size_t)*interac_blk.Dim();
      dev_ptr  += sizeof(size_t) + sizeof(size_t)*interac_blk.Dim();

      interac_cnt.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr += sizeof(size_t) + sizeof(size_t)*interac_cnt.Dim();
      dev_ptr  += sizeof(size_t) + sizeof(size_t)*interac_cnt.Dim();

      interac_mat.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr += sizeof(size_t) + sizeof(size_t)*interac_mat.Dim();
      dev_ptr  += sizeof(size_t) + sizeof(size_t)*interac_mat.Dim();

      input_perm_d.ReInit(((size_t*)data_ptr)[0],(size_t*)(dev_ptr+sizeof(size_t)),false);
      data_ptr += sizeof(size_t) + sizeof(size_t)*input_perm_d.Dim();
      dev_ptr  += sizeof(size_t) + sizeof(size_t)*input_perm_d.Dim();

      output_perm_d.ReInit(((size_t*)data_ptr)[0],(size_t*)(dev_ptr+sizeof(size_t)),false);
      data_ptr += sizeof(size_t) + sizeof(size_t)*output_perm_d.Dim();
      dev_ptr  += sizeof(size_t) + sizeof(size_t)*output_perm_d.Dim();
    }

    { // interactions
      size_t interac_indx = 0;
      size_t interac_blk_dsp = 0;
      cudaError_t error;
      for (size_t k = 0; k < interac_blk.Dim(); k++) {
        size_t vec_cnt=0;
        for(size_t j=interac_blk_dsp;j<interac_blk_dsp+interac_blk[k];j++) vec_cnt+=interac_cnt[j];
        if(vec_cnt==0){
          //interac_indx += vec_cnt;
          interac_blk_dsp += interac_blk[k];
          continue;
        }

        char *buff_in_d  =&buff[0];
        char *buff_out_d =&buff[vec_cnt*dof*M_dim0*sizeof(Real_t)];

        { // Input permutation.
          in_perm_gpu<Real_t>(&precomp_data_d[0][0], &input_data_d[0][0], buff_in_d,
                              &input_perm_d[interac_indx*4], vec_cnt, M_dim0, stream);
        }

        size_t vec_cnt0 = 0;
        for (size_t j = interac_blk_dsp; j < interac_blk_dsp + interac_blk[k];) {
          size_t vec_cnt1 = 0;
          size_t interac_mat0 = interac_mat[j];
          for (; j < interac_blk_dsp + interac_blk[k] && interac_mat[j] == interac_mat0; j++) vec_cnt1 += interac_cnt[j];
          Matrix<Real_t> M_d(M_dim0, M_dim1, (Real_t*)(precomp_data_d.dev_ptr + interac_mat0), false);
          Matrix<Real_t> Ms_d(dof*vec_cnt1, M_dim0, (Real_t*)(buff_in_d +  M_dim0*vec_cnt0*dof*sizeof(Real_t)), false);
          Matrix<Real_t> Mt_d(dof*vec_cnt1, M_dim1, (Real_t*)(buff_out_d + M_dim1*vec_cnt0*dof*sizeof(Real_t)), false);
          Matrix<Real_t>::CUBLASGEMM(Mt_d, Ms_d, M_d);
          vec_cnt0 += vec_cnt1;
        }

        { // Output permutation.
          out_perm_gpu<Real_t>(&precomp_data_d[0][0], &output_data_d[0][0], buff_out_d,
                               &output_perm_d[interac_indx*4], vec_cnt, M_dim1, stream);
        }

        interac_indx += vec_cnt;
        interac_blk_dsp += interac_blk[k];
      }
    }
  }
  Profile::Toc();

	if(SYNC) CUDA_Lock::wait();
}
#endif

template <class FMMNode>
template <int SYNC>
void FMM_Pts<FMMNode>::EvalList(SetupData<Real_t>& setup_data, bool device){
  if(setup_data.interac_data.Dim(0)==0 || setup_data.interac_data.Dim(1)==0){
    Profile::Tic("Host2Device",&this->comm,false,25);
    Profile::Toc();
    Profile::Tic("DeviceComp",&this->comm,false,20);
    Profile::Toc();
    return;
  }

#if defined(PVFMM_HAVE_CUDA)
  if (device) {
    EvalListGPU<Real_t, SYNC>(setup_data, this->dev_buffer, this->comm);
    return;
  }
#endif

  Profile::Tic("Host2Device",&this->comm,false,25);
  typename Vector<char>::Device          buff;
  typename Matrix<char>::Device  precomp_data;
  typename Matrix<char>::Device  interac_data;
  typename Matrix<Real_t>::Device  input_data;
  typename Matrix<Real_t>::Device output_data;
  if(device){
    buff        =       this-> dev_buffer. AllocDevice(false);
    precomp_data= setup_data.precomp_data->AllocDevice(false);
    interac_data= setup_data.interac_data. AllocDevice(false);
    input_data  = setup_data.  input_data->AllocDevice(false);
    output_data = setup_data. output_data->AllocDevice(false);
  }else{
    buff        =       this-> dev_buffer;
    precomp_data=*setup_data.precomp_data;
    interac_data= setup_data.interac_data;
    input_data  =*setup_data.  input_data;
    output_data =*setup_data. output_data;
  }
  Profile::Toc();

  Profile::Tic("DeviceComp",&this->comm,false,20);
  int lock_idx=-1;
  int wait_lock_idx=-1;
  if(device) wait_lock_idx=MIC_Lock::curr_lock();
  if(device) lock_idx=MIC_Lock::get_lock();
  #ifdef __INTEL_OFFLOAD
  #pragma offload if(device) target(mic:0) signal(&MIC_Lock::lock_vec[device?lock_idx:0])
  #endif
  { // Offloaded computation.

    // Set interac_data.
    size_t data_size, M_dim0, M_dim1, dof;
    Vector<size_t> interac_blk;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_mat;
    Vector<size_t>  input_perm;
    Vector<size_t> output_perm;
    { // Set interac_data.
      char* data_ptr=&interac_data[0][0];

      data_size=((size_t*)data_ptr)[0]; data_ptr+=data_size;
      data_size=((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      M_dim0   =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      M_dim1   =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      dof      =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);

      interac_blk.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+interac_blk.Dim()*sizeof(size_t);

      interac_cnt.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+interac_cnt.Dim()*sizeof(size_t);

      interac_mat.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+interac_mat.Dim()*sizeof(size_t);

      input_perm .ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+ input_perm.Dim()*sizeof(size_t);

      output_perm.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+output_perm.Dim()*sizeof(size_t);
    }

    if(device) MIC_Lock::wait_lock(wait_lock_idx);

    //Compute interaction from Chebyshev source density.
    { // interactions
      int omp_p=omp_get_max_threads();
      size_t interac_indx=0;
      size_t interac_blk_dsp=0;
      for(size_t k=0;k<interac_blk.Dim();k++){
        size_t vec_cnt=0;
        for(size_t j=interac_blk_dsp;j<interac_blk_dsp+interac_blk[k];j++) vec_cnt+=interac_cnt[j];
        if(vec_cnt==0){
          //interac_indx += vec_cnt;
          interac_blk_dsp += interac_blk[k];
          continue;
        }

        char* buff_in =&buff[0];
        char* buff_out=&buff[vec_cnt*dof*M_dim0*sizeof(Real_t)];

        // Input permutation.
        #pragma omp parallel for
        for(int tid=0;tid<omp_p;tid++){
          size_t a=( tid   *vec_cnt)/omp_p;
          size_t b=((tid+1)*vec_cnt)/omp_p;

          for(size_t i=a;i<b;i++){
            const PERM_INT_T*  perm=(PERM_INT_T*)(precomp_data[0]+input_perm[(interac_indx+i)*4+0]);
            const     Real_t*  scal=(    Real_t*)(precomp_data[0]+input_perm[(interac_indx+i)*4+1]);
            const     Real_t* v_in =(    Real_t*)(  input_data[0]+input_perm[(interac_indx+i)*4+3]);
            Real_t*           v_out=(    Real_t*)(     buff_in   +input_perm[(interac_indx+i)*4+2]);

            // TODO: Fix for dof>1
            #ifdef __MIC__
            {
              __m512d v8;
              size_t j_start=(((uintptr_t)(v_out       ) + (uintptr_t)(MEM_ALIGN-1)) & ~ (uintptr_t)(MEM_ALIGN-1))-((uintptr_t)v_out);
              size_t j_end  =(((uintptr_t)(v_out+M_dim0)                           ) & ~ (uintptr_t)(MEM_ALIGN-1))-((uintptr_t)v_out);
              j_start/=sizeof(Real_t);
              j_end  /=sizeof(Real_t);
              assert(((uintptr_t)(v_out))%sizeof(Real_t)==0);
              assert(((uintptr_t)(v_out+j_start))%64==0);
              assert(((uintptr_t)(v_out+j_end  ))%64==0);
              size_t j=0;
              for(;j<j_start;j++ ){
                v_out[j]=v_in[perm[j]]*scal[j];
              }
              for(;j<j_end  ;j+=8){
                v8=_mm512_setr_pd(
                    v_in[perm[j+0]]*scal[j+0],
                    v_in[perm[j+1]]*scal[j+1],
                    v_in[perm[j+2]]*scal[j+2],
                    v_in[perm[j+3]]*scal[j+3],
                    v_in[perm[j+4]]*scal[j+4],
                    v_in[perm[j+5]]*scal[j+5],
                    v_in[perm[j+6]]*scal[j+6],
                    v_in[perm[j+7]]*scal[j+7]);
                _mm512_storenrngo_pd(v_out+j,v8);
              }
              for(;j<M_dim0 ;j++ ){
                v_out[j]=v_in[perm[j]]*scal[j];
              }
            }
            #else
            for(size_t j=0;j<M_dim0;j++ ){
              v_out[j]=v_in[perm[j]]*scal[j];
            }
            #endif
          }
        }

        size_t vec_cnt0=0;
        for(size_t j=interac_blk_dsp;j<interac_blk_dsp+interac_blk[k];){
          size_t vec_cnt1=0;
          size_t interac_mat0=interac_mat[j];
          for(;j<interac_blk_dsp+interac_blk[k] && interac_mat[j]==interac_mat0;j++) vec_cnt1+=interac_cnt[j];
          Matrix<Real_t> M(M_dim0, M_dim1, (Real_t*)(precomp_data[0]+interac_mat0), false);
          #ifdef __MIC__
          {
            Matrix<Real_t> Ms(dof*vec_cnt1, M_dim0, (Real_t*)(buff_in +M_dim0*vec_cnt0*dof*sizeof(Real_t)), false);
            Matrix<Real_t> Mt(dof*vec_cnt1, M_dim1, (Real_t*)(buff_out+M_dim1*vec_cnt0*dof*sizeof(Real_t)), false);
            Matrix<Real_t>::GEMM(Mt,Ms,M);
          }
          #else
          #pragma omp parallel for
          for(int tid=0;tid<omp_p;tid++){
            size_t a=(dof*vec_cnt1*(tid  ))/omp_p;
            size_t b=(dof*vec_cnt1*(tid+1))/omp_p;
            Matrix<Real_t> Ms(b-a, M_dim0, (Real_t*)(buff_in +M_dim0*vec_cnt0*dof*sizeof(Real_t))+M_dim0*a, false);
            Matrix<Real_t> Mt(b-a, M_dim1, (Real_t*)(buff_out+M_dim1*vec_cnt0*dof*sizeof(Real_t))+M_dim1*a, false);
            Matrix<Real_t>::GEMM(Mt,Ms,M);
          }
          #endif
          vec_cnt0+=vec_cnt1;
        }

        // Output permutation.
        #pragma omp parallel for
        for(int tid=0;tid<omp_p;tid++){
          size_t a=( tid   *vec_cnt)/omp_p;
          size_t b=((tid+1)*vec_cnt)/omp_p;

          if(tid>      0 && a<vec_cnt){ // Find 'a' independent of other threads.
            size_t out_ptr=output_perm[(interac_indx+a)*4+3];
            if(tid>      0) while(a<vec_cnt && out_ptr==output_perm[(interac_indx+a)*4+3]) a++;
          }
          if(tid<omp_p-1 && b<vec_cnt){ // Find 'b' independent of other threads.
            size_t out_ptr=output_perm[(interac_indx+b)*4+3];
            if(tid<omp_p-1) while(b<vec_cnt && out_ptr==output_perm[(interac_indx+b)*4+3]) b++;
          }
          for(size_t i=a;i<b;i++){ // Compute permutations.
            const PERM_INT_T*  perm=(PERM_INT_T*)(precomp_data[0]+output_perm[(interac_indx+i)*4+0]);
            const     Real_t*  scal=(    Real_t*)(precomp_data[0]+output_perm[(interac_indx+i)*4+1]);
            const     Real_t* v_in =(    Real_t*)(    buff_out   +output_perm[(interac_indx+i)*4+2]);
            Real_t*           v_out=(    Real_t*)( output_data[0]+output_perm[(interac_indx+i)*4+3]);

            // TODO: Fix for dof>1
            #ifdef __MIC__
            {
              __m512d v8;
              __m512d v_old;
              size_t j_start=(((uintptr_t)(v_out       ) + (uintptr_t)(MEM_ALIGN-1)) & ~ (uintptr_t)(MEM_ALIGN-1))-((uintptr_t)v_out);
              size_t j_end  =(((uintptr_t)(v_out+M_dim1)                           ) & ~ (uintptr_t)(MEM_ALIGN-1))-((uintptr_t)v_out);
              j_start/=sizeof(Real_t);
              j_end  /=sizeof(Real_t);
              assert(((uintptr_t)(v_out))%sizeof(Real_t)==0);
              assert(((uintptr_t)(v_out+j_start))%64==0);
              assert(((uintptr_t)(v_out+j_end  ))%64==0);
              size_t j=0;
              for(;j<j_start;j++ ){
                v_out[j]+=v_in[perm[j]]*scal[j];
              }
              for(;j<j_end  ;j+=8){
                v_old=_mm512_load_pd(v_out+j);
                v8=_mm512_setr_pd(
                    v_in[perm[j+0]]*scal[j+0],
                    v_in[perm[j+1]]*scal[j+1],
                    v_in[perm[j+2]]*scal[j+2],
                    v_in[perm[j+3]]*scal[j+3],
                    v_in[perm[j+4]]*scal[j+4],
                    v_in[perm[j+5]]*scal[j+5],
                    v_in[perm[j+6]]*scal[j+6],
                    v_in[perm[j+7]]*scal[j+7]);
                v_old=_mm512_add_pd(v_old, v8);
                _mm512_storenrngo_pd(v_out+j,v_old);
              }
              for(;j<M_dim1 ;j++ ){
                v_out[j]+=v_in[perm[j]]*scal[j];
              }
            }
            #else
            for(size_t j=0;j<M_dim1;j++ ){
              v_out[j]+=v_in[perm[j]]*scal[j];
            }
            #endif
          }
        }

        interac_indx+=vec_cnt;
        interac_blk_dsp+=interac_blk[k];
      }
    }

    if(device) MIC_Lock::release_lock(lock_idx);
  }

  #ifdef __INTEL_OFFLOAD
  if(SYNC){
    #pragma offload if(device) target(mic:0)
    {if(device) MIC_Lock::wait_lock(lock_idx);}
  }
  #endif

  Profile::Toc();
}



template <class FMMNode>
void FMM_Pts<FMMNode>::Source2UpSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_s2m;
    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[0];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[0];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->Depth()==level || level==-1) && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim()) && nodes_in [i]->IsLeaf() && !nodes_in [i]->IsGhost()) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->Depth()==level || level==-1) && (nodes_out[i]->src_coord.Dim() || nodes_out[i]->surf_coord.Dim()) && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=((FMMNode*)nodes[i])->src_coord;
      Vector<Real_t>& value_vec=((FMMNode*)nodes[i])->src_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode*)nodes[i])->surf_coord;
      Vector<Real_t>& value_vec=((FMMNode*)nodes[i])->surf_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=tree->upwd_check_surf[((FMMNode*)nodes[i])->Depth()];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode*)nodes[i])->FMMData())->upward_equiv;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);
    if(this->ScaleInvar()){ // Set scal
      const Kernel<Real_t>* ker=kernel->k_m2m;
      for(size_t l=0;l<MAX_DEPTH;l++){ // scal[l*4+2]
        Vector<Real_t>& scal=data.interac_data.scal[l*4+2];
        Vector<Real_t>& scal_exp=ker->trg_scal;
        scal.ReInit(scal_exp.Dim());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
        }
      }
      for(size_t l=0;l<MAX_DEPTH;l++){ // scal[l*4+3]
        Vector<Real_t>& scal=data.interac_data.scal[l*4+3];
        Vector<Real_t>& scal_exp=ker->src_scal;
        scal.ReInit(scal_exp.Dim());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
        }
      }
    }

    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid];

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->Depth());

        size_t interac_cnt_=0;
        { // S2U_Type
          Mat_Type type=S2U_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(scoord[0]+0.5*s)+(0+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(scoord[1]+0.5*s)+(0+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(scoord[2]+0.5*s)+(0+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
    { // Set M[2], M[3]
      InteracData& interac_data=data.interac_data;
      pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
      pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
      if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
        data.interac_data.M[2]=this->mat->Mat(level, UC2UE0_Type, 0);
        data.interac_data.M[3]=this->mat->Mat(level, UC2UE1_Type, 0);
      }else{
        data.interac_data.M[2].ReInit(0,0);
        data.interac_data.M[3].ReInit(0,0);
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::Source2Up(SetupData<Real_t>&  setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add Source2Up contribution.
  this->EvalListPts(setup_data, device);
}


template <class FMMNode>
void FMM_Pts<FMMNode>::Up2UpSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=kernel->k_m2m;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=U2U_Type;

    setup_data. input_data=&buff[0];
    setup_data.output_data=&buff[0];
    Vector<FMMNode_t*>& nodes_in =n_list[0];
    Vector<FMMNode_t*>& nodes_out=n_list[0];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->Depth()==level+1) && nodes_in [i]->pt_cnt[0]) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->Depth()==level  ) && nodes_out[i]->pt_cnt[0]) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode*)nodes_in [i])->FMMData())->upward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)nodes_out[i])->FMMData())->upward_equiv);

  SetupInterac(setup_data,device);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::Up2Up     (SetupData<Real_t>& setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add Up2Up contribution.
  EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Pts<FMMNode>::PeriodicBC(FMMNode* node){
  if(!this->ScaleInvar() || this->MultipoleOrder()==0) return;
  Matrix<Real_t>& M = Precomp(0, BC_Type, 0);

  assert(node->FMMData()->upward_equiv.Dim()>0);
  int dof=1;

  Vector<Real_t>& upward_equiv=node->FMMData()->upward_equiv;
  Vector<Real_t>& dnward_equiv=node->FMMData()->dnward_equiv;
  assert(upward_equiv.Dim()==M.Dim(0)*dof);
  assert(dnward_equiv.Dim()==M.Dim(1)*dof);
  Matrix<Real_t> d_equiv(dof,M.Dim(1),&dnward_equiv[0],false);
  Matrix<Real_t> u_equiv(dof,M.Dim(0),&upward_equiv[0],false);
  Matrix<Real_t>::GEMM(d_equiv,u_equiv,M);
}



template <class FMMNode>
void FMM_Pts<FMMNode>::FFT_UpEquiv(size_t dof, size_t m, size_t ker_dim0, Vector<size_t>& fft_vec, Vector<Real_t>& fft_scal,
    Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_){

  size_t n1=m*2;
  size_t n2=n1*n1;
  size_t n3=n1*n2;
  size_t n3_=n2*(n1/2+1);
  size_t chld_cnt=1UL<<COORD_DIM;
  size_t fftsize_in =2*n3_*chld_cnt*ker_dim0*dof;
  int omp_p=omp_get_max_threads();

  //Load permutation map.
  size_t n=6*(m-1)*(m-1)+2;
  static Vector<size_t> map;
  { // Build map to reorder upward_equiv
    size_t n_old=map.Dim();
    if(n_old!=n){
      Real_t c[3]={0,0,0};
      Vector<Real_t> surf=surface(m, c, (Real_t)(m-1), 0);
      map.Resize(surf.Dim()/COORD_DIM);
      for(size_t i=0;i<map.Dim();i++)
        map[i]=((size_t)(m-1-surf[i*3]+0.5))+((size_t)(m-1-surf[i*3+1]+0.5))*n1+((size_t)(m-1-surf[i*3+2]+0.5))*n2;
    }
  }
  { // Build FFTW plan.
    if(!vlist_fft_flag){
      int nnn[3]={(int)n1,(int)n1,(int)n1};
      void *fftw_in, *fftw_out;
      fftw_in  = mem::aligned_new<Real_t>(  n3 *ker_dim0*chld_cnt);
      fftw_out = mem::aligned_new<Real_t>(2*n3_*ker_dim0*chld_cnt);
      vlist_fftplan = FFTW_t<Real_t>::fft_plan_many_dft_r2c(COORD_DIM,nnn,ker_dim0*chld_cnt,
          (Real_t*)fftw_in, NULL, 1, n3, (typename FFTW_t<Real_t>::cplx*)(fftw_out),NULL, 1, n3_);
      mem::aligned_delete<Real_t>((Real_t*)fftw_in );
      mem::aligned_delete<Real_t>((Real_t*)fftw_out);
      vlist_fft_flag=true;
    }
  }

  { // Offload section
    size_t n_in = fft_vec.Dim();
    #pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++){
      size_t node_start=(n_in*(pid  ))/omp_p;
      size_t node_end  =(n_in*(pid+1))/omp_p;
      Vector<Real_t> buffer(fftsize_in, &buffer_[fftsize_in*pid], false);
      for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
        Matrix<Real_t>  upward_equiv(chld_cnt,n*ker_dim0*dof,&input_data[0] + fft_vec[node_idx],false);
        Vector<Real_t> upward_equiv_fft(fftsize_in, &output_data[fftsize_in *node_idx], false);
        upward_equiv_fft.SetZero();

        // Rearrange upward equivalent data.
        for(size_t k=0;k<n;k++){
          size_t idx=map[k];
          for(int j1=0;j1<dof;j1++)
          for(int j0=0;j0<(int)chld_cnt;j0++)
          for(int i=0;i<ker_dim0;i++)
            upward_equiv_fft[idx+(j0+(i+j1*ker_dim0)*chld_cnt)*n3]=upward_equiv[j0][ker_dim0*(n*j1+k)+i]*fft_scal[ker_dim0*node_idx+i];
        }

        // Compute FFT.
        for(int i=0;i<dof;i++)
          FFTW_t<Real_t>::fft_execute_dft_r2c(vlist_fftplan, (Real_t*)&upward_equiv_fft[i*  n3 *ker_dim0*chld_cnt],
                                      (typename FFTW_t<Real_t>::cplx*)&buffer          [i*2*n3_*ker_dim0*chld_cnt]);

        //Compute flops.
        #ifndef FFTW3_MKL
        double add, mul, fma;
        FFTW_t<Real_t>::fftw_flops(vlist_fftplan, &add, &mul, &fma);
        #ifndef __INTEL_OFFLOAD0
        Profile::Add_FLOP((long long)(add+mul+2*fma));
        #endif
        #endif

        for(int i=0;i<ker_dim0*dof;i++)
        for(size_t j=0;j<n3_;j++)
        for(size_t k=0;k<chld_cnt;k++){
          upward_equiv_fft[2*(chld_cnt*(n3_*i+j)+k)+0]=buffer[2*(n3_*(chld_cnt*i+k)+j)+0];
          upward_equiv_fft[2*(chld_cnt*(n3_*i+j)+k)+1]=buffer[2*(n3_*(chld_cnt*i+k)+j)+1];
        }
      }
    }
  }
}

template <class FMMNode>
void FMM_Pts<FMMNode>::FFT_Check2Equiv(size_t dof, size_t m, size_t ker_dim1, Vector<size_t>& ifft_vec, Vector<Real_t>& ifft_scal,
    Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_){

  size_t n1=m*2;
  size_t n2=n1*n1;
  size_t n3=n1*n2;
  size_t n3_=n2*(n1/2+1);
  size_t chld_cnt=1UL<<COORD_DIM;
  size_t fftsize_out=2*n3_*dof*ker_dim1*chld_cnt;
  int omp_p=omp_get_max_threads();

  //Load permutation map.
  size_t n=6*(m-1)*(m-1)+2;
  static Vector<size_t> map;
  { // Build map to reorder dnward_check
    size_t n_old=map.Dim();
    if(n_old!=n){
      Real_t c[3]={0,0,0};
      Vector<Real_t> surf=surface(m, c, (Real_t)(m-1), 0);
      map.Resize(surf.Dim()/COORD_DIM);
      for(size_t i=0;i<map.Dim();i++)
        map[i]=((size_t)(m*2-0.5-surf[i*3]))+((size_t)(m*2-0.5-surf[i*3+1]))*n1+((size_t)(m*2-0.5-surf[i*3+2]))*n2;
      //map;//.AllocDevice(true);
    }
  }
  { // Build FFTW plan.
    if(!vlist_ifft_flag){
      //Build FFTW plan.
      int nnn[3]={(int)n1,(int)n1,(int)n1};
      Real_t *fftw_in, *fftw_out;
      fftw_in  = mem::aligned_new<Real_t>(2*n3_*ker_dim1*chld_cnt);
      fftw_out = mem::aligned_new<Real_t>(  n3 *ker_dim1*chld_cnt);
      vlist_ifftplan = FFTW_t<Real_t>::fft_plan_many_dft_c2r(COORD_DIM,nnn,ker_dim1*chld_cnt,
          (typename FFTW_t<Real_t>::cplx*)fftw_in, NULL, 1, n3_, (Real_t*)(fftw_out),NULL, 1, n3);
      mem::aligned_delete<Real_t>(fftw_in);
      mem::aligned_delete<Real_t>(fftw_out);
      vlist_ifft_flag=true;
    }
  }

  { // Offload section
    assert(buffer_.Dim()>=2*fftsize_out*omp_p);
    size_t n_out=ifft_vec.Dim();
    #pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++){
      size_t node_start=(n_out*(pid  ))/omp_p;
      size_t node_end  =(n_out*(pid+1))/omp_p;
      Vector<Real_t> buffer0(fftsize_out, &buffer_[fftsize_out*(2*pid+0)], false);
      Vector<Real_t> buffer1(fftsize_out, &buffer_[fftsize_out*(2*pid+1)], false);
      for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
        Vector<Real_t> dnward_check_fft(fftsize_out, &input_data[fftsize_out*node_idx], false);
        Vector<Real_t> dnward_equiv(ker_dim1*n*dof*chld_cnt,&output_data[0] + ifft_vec[node_idx],false);

        //De-interleave data.
        for(int i=0;i<ker_dim1*dof;i++)
        for(size_t j=0;j<n3_;j++)
        for(size_t k=0;k<chld_cnt;k++){
          buffer0[2*(n3_*(ker_dim1*dof*k+i)+j)+0]=dnward_check_fft[2*(chld_cnt*(n3_*i+j)+k)+0];
          buffer0[2*(n3_*(ker_dim1*dof*k+i)+j)+1]=dnward_check_fft[2*(chld_cnt*(n3_*i+j)+k)+1];
        }

        // Compute FFT.
        for(int i=0;i<dof;i++)
          FFTW_t<Real_t>::fft_execute_dft_c2r(vlist_ifftplan, (typename FFTW_t<Real_t>::cplx*)&buffer0[i*2*n3_*ker_dim1*chld_cnt],
                                                                                     (Real_t*)&buffer1[i*  n3 *ker_dim1*chld_cnt]);
        //Compute flops.
        #ifndef FFTW3_MKL
        double add, mul, fma;
        FFTW_t<Real_t>::fftw_flops(vlist_ifftplan, &add, &mul, &fma);
        #ifndef __INTEL_OFFLOAD0
        Profile::Add_FLOP((long long)(add+mul+2*fma)*dof);
        #endif
        #endif

        // Rearrange downward check data.
        for(size_t k=0;k<n;k++){
          size_t idx=map[k];
          for(int j1=0;j1<dof;j1++)
          for(int j0=0;j0<(int)chld_cnt;j0++)
          for(int i=0;i<ker_dim1;i++)
            dnward_equiv[ker_dim1*(n*(dof*j0+j1)+k)+i]+=buffer1[idx+(i+(j1+j0*dof)*ker_dim1)*n3]*ifft_scal[ker_dim1*node_idx+i];
        }
      }
    }
  }
}

template<class Real_t>
inline void matmult_8x8x2(Real_t*& M_, Real_t*& IN0, Real_t*& IN1, Real_t*& OUT0, Real_t*& OUT1){
  // Generic code.
  Real_t out_reg000, out_reg001, out_reg010, out_reg011;
  Real_t out_reg100, out_reg101, out_reg110, out_reg111;
  Real_t  in_reg000,  in_reg001,  in_reg010,  in_reg011;
  Real_t  in_reg100,  in_reg101,  in_reg110,  in_reg111;
  Real_t   m_reg000,   m_reg001,   m_reg010,   m_reg011;
  Real_t   m_reg100,   m_reg101,   m_reg110,   m_reg111;
  //#pragma unroll
  for(int i1=0;i1<8;i1+=2){
    Real_t* IN0_=IN0;
    Real_t* IN1_=IN1;

    out_reg000=OUT0[ 0]; out_reg001=OUT0[ 1];
    out_reg010=OUT0[ 2]; out_reg011=OUT0[ 3];
    out_reg100=OUT1[ 0]; out_reg101=OUT1[ 1];
    out_reg110=OUT1[ 2]; out_reg111=OUT1[ 3];
    //#pragma unroll
    for(int i2=0;i2<8;i2+=2){
      m_reg000=M_[ 0]; m_reg001=M_[ 1];
      m_reg010=M_[ 2]; m_reg011=M_[ 3];
      m_reg100=M_[16]; m_reg101=M_[17];
      m_reg110=M_[18]; m_reg111=M_[19];

      in_reg000=IN0_[0]; in_reg001=IN0_[1];
      in_reg010=IN0_[2]; in_reg011=IN0_[3];
      in_reg100=IN1_[0]; in_reg101=IN1_[1];
      in_reg110=IN1_[2]; in_reg111=IN1_[3];

      out_reg000 += m_reg000*in_reg000 - m_reg001*in_reg001;
      out_reg001 += m_reg000*in_reg001 + m_reg001*in_reg000;
      out_reg010 += m_reg010*in_reg000 - m_reg011*in_reg001;
      out_reg011 += m_reg010*in_reg001 + m_reg011*in_reg000;

      out_reg000 += m_reg100*in_reg010 - m_reg101*in_reg011;
      out_reg001 += m_reg100*in_reg011 + m_reg101*in_reg010;
      out_reg010 += m_reg110*in_reg010 - m_reg111*in_reg011;
      out_reg011 += m_reg110*in_reg011 + m_reg111*in_reg010;

      out_reg100 += m_reg000*in_reg100 - m_reg001*in_reg101;
      out_reg101 += m_reg000*in_reg101 + m_reg001*in_reg100;
      out_reg110 += m_reg010*in_reg100 - m_reg011*in_reg101;
      out_reg111 += m_reg010*in_reg101 + m_reg011*in_reg100;

      out_reg100 += m_reg100*in_reg110 - m_reg101*in_reg111;
      out_reg101 += m_reg100*in_reg111 + m_reg101*in_reg110;
      out_reg110 += m_reg110*in_reg110 - m_reg111*in_reg111;
      out_reg111 += m_reg110*in_reg111 + m_reg111*in_reg110;

      M_+=32; // Jump to (column+2).
      IN0_+=4;
      IN1_+=4;
    }
    OUT0[ 0]=out_reg000; OUT0[ 1]=out_reg001;
    OUT0[ 2]=out_reg010; OUT0[ 3]=out_reg011;
    OUT1[ 0]=out_reg100; OUT1[ 1]=out_reg101;
    OUT1[ 2]=out_reg110; OUT1[ 3]=out_reg111;
    M_+=4-64*2; // Jump back to first column (row+2).
    OUT0+=4;
    OUT1+=4;
  }
}

#if defined(__AVX__) || defined(__SSE3__)
template<>
inline void matmult_8x8x2<double>(double*& M_, double*& IN0, double*& IN1, double*& OUT0, double*& OUT1){
#ifdef __AVX__ //AVX code.
  __m256d out00,out01,out10,out11;
  __m256d out20,out21,out30,out31;
  double* in0__ = IN0;
  double* in1__ = IN1;

  out00 = _mm256_load_pd(OUT0);
  out01 = _mm256_load_pd(OUT1);
  out10 = _mm256_load_pd(OUT0+4);
  out11 = _mm256_load_pd(OUT1+4);
  out20 = _mm256_load_pd(OUT0+8);
  out21 = _mm256_load_pd(OUT1+8);
  out30 = _mm256_load_pd(OUT0+12);
  out31 = _mm256_load_pd(OUT1+12);
  for(int i2=0;i2<8;i2+=2){
    __m256d m00;
    __m256d ot00;
    __m256d mt0,mtt0;
    __m256d in00,in00_r,in01,in01_r;
    in00 = _mm256_broadcast_pd((const __m128d*)in0__);
    in00_r = _mm256_permute_pd(in00,5);
    in01 = _mm256_broadcast_pd((const __m128d*)in1__);
    in01_r = _mm256_permute_pd(in01,5);

    m00 = _mm256_load_pd(M_);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));

    m00 = _mm256_load_pd(M_+4);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));

    m00 = _mm256_load_pd(M_+8);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));

    m00 = _mm256_load_pd(M_+12);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));


    in00 = _mm256_broadcast_pd((const __m128d*) (in0__+2));
    in00_r = _mm256_permute_pd(in00,5);
    in01 = _mm256_broadcast_pd((const __m128d*) (in1__+2));
    in01_r = _mm256_permute_pd(in01,5);

    m00 = _mm256_load_pd(M_+16);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));

    m00 = _mm256_load_pd(M_+20);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));

    m00 = _mm256_load_pd(M_+24);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));

    m00 = _mm256_load_pd(M_+28);

    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));

    ot00 = _mm256_mul_pd(mt0,in01);
    out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));

    M_ += 32;
    in0__ += 4;
    in1__ += 4;
  }
  _mm256_store_pd(OUT0,out00);
  _mm256_store_pd(OUT1,out01);
  _mm256_store_pd(OUT0+4,out10);
  _mm256_store_pd(OUT1+4,out11);
  _mm256_store_pd(OUT0+8,out20);
  _mm256_store_pd(OUT1+8,out21);
  _mm256_store_pd(OUT0+12,out30);
  _mm256_store_pd(OUT1+12,out31);
#elif defined __SSE3__ // SSE code.
  __m128d out00, out01, out10, out11;
  __m128d in00, in01, in10, in11;
  __m128d m00, m01, m10, m11;
  //#pragma unroll
  for(int i1=0;i1<8;i1+=2){
    double* IN0_=IN0;
    double* IN1_=IN1;

    out00 =_mm_load_pd (OUT0  );
    out10 =_mm_load_pd (OUT0+2);
    out01 =_mm_load_pd (OUT1  );
    out11 =_mm_load_pd (OUT1+2);
    //#pragma unroll
    for(int i2=0;i2<8;i2+=2){
      m00 =_mm_load1_pd (M_   );
      m10 =_mm_load1_pd (M_+ 2);
      m01 =_mm_load1_pd (M_+16);
      m11 =_mm_load1_pd (M_+18);

      in00 =_mm_load_pd (IN0_  );
      in10 =_mm_load_pd (IN0_+2);
      in01 =_mm_load_pd (IN1_  );
      in11 =_mm_load_pd (IN1_+2);

      out00 = _mm_add_pd   (out00, _mm_mul_pd(m00 , in00 ));
      out00 = _mm_add_pd   (out00, _mm_mul_pd(m01 , in10 ));
      out01 = _mm_add_pd   (out01, _mm_mul_pd(m00 , in01 ));
      out01 = _mm_add_pd   (out01, _mm_mul_pd(m01 , in11 ));
      out10 = _mm_add_pd   (out10, _mm_mul_pd(m10 , in00 ));
      out10 = _mm_add_pd   (out10, _mm_mul_pd(m11 , in10 ));
      out11 = _mm_add_pd   (out11, _mm_mul_pd(m10 , in01 ));
      out11 = _mm_add_pd   (out11, _mm_mul_pd(m11 , in11 ));


      m00 =_mm_load1_pd (M_+   1);
      m10 =_mm_load1_pd (M_+ 2+1);
      m01 =_mm_load1_pd (M_+16+1);
      m11 =_mm_load1_pd (M_+18+1);
      in00 =_mm_shuffle_pd (in00,in00,_MM_SHUFFLE2(0,1));
      in01 =_mm_shuffle_pd (in01,in01,_MM_SHUFFLE2(0,1));
      in10 =_mm_shuffle_pd (in10,in10,_MM_SHUFFLE2(0,1));
      in11 =_mm_shuffle_pd (in11,in11,_MM_SHUFFLE2(0,1));
      out00 = _mm_addsub_pd(out00, _mm_mul_pd(m00, in00));
      out00 = _mm_addsub_pd(out00, _mm_mul_pd(m01, in10));
      out01 = _mm_addsub_pd(out01, _mm_mul_pd(m00, in01));
      out01 = _mm_addsub_pd(out01, _mm_mul_pd(m01, in11));
      out10 = _mm_addsub_pd(out10, _mm_mul_pd(m10, in00));
      out10 = _mm_addsub_pd(out10, _mm_mul_pd(m11, in10));
      out11 = _mm_addsub_pd(out11, _mm_mul_pd(m10, in01));
      out11 = _mm_addsub_pd(out11, _mm_mul_pd(m11, in11));

      M_+=32; // Jump to (column+2).
      IN0_+=4;
      IN1_+=4;
    }
    _mm_store_pd (OUT0  ,out00);
    _mm_store_pd (OUT0+2,out10);
    _mm_store_pd (OUT1  ,out01);
    _mm_store_pd (OUT1+2,out11);
    M_+=4-64*2; // Jump back to first column (row+2).
    OUT0+=4;
    OUT1+=4;
  }
#endif
}
#endif

#if defined(__SSE3__)
template<>
inline void matmult_8x8x2<float>(float*& M_, float*& IN0, float*& IN1, float*& OUT0, float*& OUT1){
#if defined __SSE3__ // SSE code.
  __m128 out00,out01,out10,out11;
  __m128 out20,out21,out30,out31;
  float* in0__ = IN0;
  float* in1__ = IN1;

  out00 = _mm_load_ps(OUT0);
  out01 = _mm_load_ps(OUT1);
  out10 = _mm_load_ps(OUT0+4);
  out11 = _mm_load_ps(OUT1+4);
  out20 = _mm_load_ps(OUT0+8);
  out21 = _mm_load_ps(OUT1+8);
  out30 = _mm_load_ps(OUT0+12);
  out31 = _mm_load_ps(OUT1+12);
  for(int i2=0;i2<8;i2+=2){
    __m128 m00;
    __m128 mt0,mtt0;
    __m128 in00,in00_r,in01,in01_r;



    in00 = _mm_castpd_ps(_mm_load_pd1((const double*)in0__));
    in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
    in01 = _mm_castpd_ps(_mm_load_pd1((const double*)in1__));
    in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));

    m00 = _mm_load_ps(M_);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));

    out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
    out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));

    m00 = _mm_load_ps(M_+4);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));

    out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01  ));
    out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));

    m00 = _mm_load_ps(M_+8);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));

    out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
    out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));

    m00 = _mm_load_ps(M_+12);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,  in00));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));

    out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
    out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));



    in00 = _mm_castpd_ps(_mm_load_pd1((const double*) (in0__+2)));
    in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
    in01 = _mm_castpd_ps(_mm_load_pd1((const double*) (in1__+2)));
    in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));

    m00 = _mm_load_ps(M_+16);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));

    out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
    out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));

    m00 = _mm_load_ps(M_+20);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));

    out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01 ));
    out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));

    m00 = _mm_load_ps(M_+24);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));

    out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
    out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));

    m00 = _mm_load_ps(M_+28);

    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));

    out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
    out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));

    M_ += 32;
    in0__ += 4;
    in1__ += 4;
  }
  _mm_store_ps(OUT0,out00);
  _mm_store_ps(OUT1,out01);
  _mm_store_ps(OUT0+4,out10);
  _mm_store_ps(OUT1+4,out11);
  _mm_store_ps(OUT0+8,out20);
  _mm_store_ps(OUT1+8,out21);
  _mm_store_ps(OUT0+12,out30);
  _mm_store_ps(OUT1+12,out31);
#endif
}
#endif

template <class Real_t>
void VListHadamard(size_t dof, size_t M_dim, size_t ker_dim0, size_t ker_dim1, Vector<size_t>& interac_dsp,
    Vector<size_t>& interac_vec, Vector<Real_t*>& precomp_mat, Vector<Real_t>& fft_in, Vector<Real_t>& fft_out){

  size_t chld_cnt=1UL<<COORD_DIM;
  size_t fftsize_in =M_dim*ker_dim0*chld_cnt*2;
  size_t fftsize_out=M_dim*ker_dim1*chld_cnt*2;
  Real_t* zero_vec0=mem::aligned_new<Real_t>(fftsize_in );
  Real_t* zero_vec1=mem::aligned_new<Real_t>(fftsize_out);
  size_t n_out=fft_out.Dim()/fftsize_out;

  // Set buff_out to zero.
  #pragma omp parallel for
  for(size_t k=0;k<n_out;k++){
    Vector<Real_t> dnward_check_fft(fftsize_out, &fft_out[k*fftsize_out], false);
    dnward_check_fft.SetZero();
  }

  // Build list of interaction pairs (in, out vectors).
  size_t mat_cnt=precomp_mat.Dim();
  size_t blk1_cnt=interac_dsp.Dim()/mat_cnt;
  const size_t V_BLK_SIZE=V_BLK_CACHE*64/sizeof(Real_t);
  Real_t** IN_ =mem::aligned_new<Real_t*>(2*V_BLK_SIZE*blk1_cnt*mat_cnt);
  Real_t** OUT_=mem::aligned_new<Real_t*>(2*V_BLK_SIZE*blk1_cnt*mat_cnt);
  #pragma omp parallel for
  for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++){
    size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
    size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
    size_t interac_cnt  = interac_dsp1-interac_dsp0;
    for(size_t j=0;j<interac_cnt;j++){
      IN_ [2*V_BLK_SIZE*interac_blk1 +j]=&fft_in [interac_vec[(interac_dsp0+j)*2+0]];
      OUT_[2*V_BLK_SIZE*interac_blk1 +j]=&fft_out[interac_vec[(interac_dsp0+j)*2+1]];
    }
    IN_ [2*V_BLK_SIZE*interac_blk1 +interac_cnt]=zero_vec0;
    OUT_[2*V_BLK_SIZE*interac_blk1 +interac_cnt]=zero_vec1;
  }

  int omp_p=omp_get_max_threads();
  #pragma omp parallel for
  for(int pid=0; pid<omp_p; pid++){
    size_t a=( pid   *M_dim)/omp_p;
    size_t b=((pid+1)*M_dim)/omp_p;

    for(int in_dim=0;in_dim<ker_dim0;in_dim++)
    for(int ot_dim=0;ot_dim<ker_dim1;ot_dim++)
    for(size_t     blk1=0;     blk1<blk1_cnt;    blk1++)
    for(size_t        k=a;        k<       b;       k++)
    for(size_t mat_indx=0; mat_indx< mat_cnt;mat_indx++){
      size_t interac_blk1 = blk1*mat_cnt+mat_indx;
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;

      Real_t** IN = IN_ + 2*V_BLK_SIZE*interac_blk1;
      Real_t** OUT= OUT_+ 2*V_BLK_SIZE*interac_blk1;

      Real_t* M = precomp_mat[mat_indx] + k*chld_cnt*chld_cnt*2 + (ot_dim+in_dim*ker_dim1)*M_dim*128;
      {
        for(size_t j=0;j<interac_cnt;j+=2){
          Real_t* M_   = M;
          Real_t* IN0  = IN [j+0] + (in_dim*M_dim+k)*chld_cnt*2;
          Real_t* IN1  = IN [j+1] + (in_dim*M_dim+k)*chld_cnt*2;
          Real_t* OUT0 = OUT[j+0] + (ot_dim*M_dim+k)*chld_cnt*2;
          Real_t* OUT1 = OUT[j+1] + (ot_dim*M_dim+k)*chld_cnt*2;

#ifdef __SSE__
          if (j+2 < interac_cnt) { // Prefetch
            _mm_prefetch(((char *)(IN[j+2] + (in_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+2] + (in_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + (in_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + (in_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);

            _mm_prefetch(((char *)(OUT[j+2] + (ot_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+2] + (ot_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + (ot_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + (ot_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
          }
#endif

          matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
        }
      }
    }
  }

  // Compute flops.
  {
    Profile::Add_FLOP(8*8*8*(interac_vec.Dim()/2)*M_dim*ker_dim0*ker_dim1*dof);
  }

  // Free memory
  mem::aligned_delete<Real_t*>(IN_ );
  mem::aligned_delete<Real_t*>(OUT_);
  mem::aligned_delete<Real_t>(zero_vec0);
  mem::aligned_delete<Real_t>(zero_vec1);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::V_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  if(level==0) return;
  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=kernel->k_m2l;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=V1_Type;

    setup_data. input_data=&buff[0];
    setup_data.output_data=&buff[1];
    Vector<FMMNode_t*>& nodes_in =n_list[2];
    Vector<FMMNode_t*>& nodes_out=n_list[3];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->Depth()==level-1 || level==-1) && nodes_in [i]->pt_cnt[0]) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->Depth()==level-1 || level==-1) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
  }
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode*)((FMMNode*)nodes_in [i])->Child(0))->FMMData())->upward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)((FMMNode*)nodes_out[i])->Child(0))->FMMData())->dnward_equiv);

  /////////////////////////////////////////////////////////////////////////////
  Real_t eps=1e-10;

  size_t n_in =nodes_in .size();
  size_t n_out=nodes_out.size();

  // Setup precomputed data.
  //if(setup_data.precomp_data->Dim(0)*setup_data.precomp_data->Dim(1)==0) SetupPrecomp(setup_data,device);

  // Build interac_data
  Profile::Tic("Interac-Data",&this->comm,true,25);
  Matrix<char>& interac_data=setup_data.interac_data;
  if(n_out>0 && n_in >0){ // Build precomp_data, interac_data

    size_t precomp_offset=0;
    Mat_Type& interac_type=setup_data.interac_type[0];
    size_t mat_cnt=this->interac_list.ListCount(interac_type);
    Matrix<size_t> precomp_data_offset;
    std::vector<size_t> interac_mat;
    std::vector<Real_t*> interac_mat_ptr;
#if 0 // Since we skip SetupPrecomp for V-list
    { // Load precomp_data for interac_type.
      struct HeaderData{
        size_t total_size;
        size_t      level;
        size_t   mat_cnt ;
        size_t  max_depth;
      };
      Matrix<char>& precomp_data=*setup_data.precomp_data;
      char* indx_ptr=precomp_data[0]+precomp_offset;
      HeaderData& header=*(HeaderData*)indx_ptr;indx_ptr+=sizeof(HeaderData);
      precomp_data_offset.ReInit(header.mat_cnt,1+(2+2)*header.max_depth, (size_t*)indx_ptr, false);
      precomp_offset+=header.total_size;
      for(size_t mat_id=0;mat_id<mat_cnt;mat_id++){
        Matrix<Real_t>& M0 = this->mat->Mat(level, interac_type, mat_id);
        assert(M0.Dim(0)>0 && M0.Dim(1)>0); UNUSED(M0);
        interac_mat.push_back(precomp_data_offset[mat_id][0]);
      }
    }
#else
    {
      for(size_t mat_id=0;mat_id<mat_cnt;mat_id++){
        Matrix<Real_t>& M = this->mat->Mat(level, interac_type, mat_id);
        interac_mat_ptr.push_back(&M[0][0]);
      }
    }
#endif

    size_t dof;
    size_t m=MultipoleOrder();
    size_t ker_dim0=setup_data.kernel->ker_dim[0];
    size_t ker_dim1=setup_data.kernel->ker_dim[1];
    size_t fftsize;
    {
      size_t n1=m*2;
      size_t n2=n1*n1;
      size_t n3_=n2*(n1/2+1);
      size_t chld_cnt=1UL<<COORD_DIM;
      fftsize=2*n3_*chld_cnt;
      dof=1;
    }

    int omp_p=omp_get_max_threads();
    size_t buff_size=DEVICE_BUFFER_SIZE*1024l*1024l;
    size_t n_blk0=2*fftsize*dof*(ker_dim0*n_in +ker_dim1*n_out)*sizeof(Real_t)/buff_size;
    if(n_blk0==0) n_blk0=1;

    std::vector<std::vector<size_t> >  fft_vec(n_blk0);
    std::vector<std::vector<size_t> > ifft_vec(n_blk0);
    std::vector<std::vector<Real_t> >  fft_scl(n_blk0);
    std::vector<std::vector<Real_t> > ifft_scl(n_blk0);
    std::vector<std::vector<size_t> > interac_vec(n_blk0);
    std::vector<std::vector<size_t> > interac_dsp(n_blk0);
    {
      Matrix<Real_t>&  input_data=*setup_data. input_data;
      Matrix<Real_t>& output_data=*setup_data.output_data;
      std::vector<std::vector<FMMNode*> > nodes_blk_in (n_blk0);
      std::vector<std::vector<FMMNode*> > nodes_blk_out(n_blk0);

      Vector<Real_t> src_scal=this->kernel->k_m2l->src_scal;
      Vector<Real_t> trg_scal=this->kernel->k_m2l->trg_scal;

      for(size_t i=0;i<n_in;i++) ((FMMNode*)nodes_in[i])->node_id=i;
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        size_t blk0_start=(n_out* blk0   )/n_blk0;
        size_t blk0_end  =(n_out*(blk0+1))/n_blk0;

        std::vector<FMMNode*>& nodes_in_ =nodes_blk_in [blk0];
        std::vector<FMMNode*>& nodes_out_=nodes_blk_out[blk0];
        { // Build node list for blk0.
          std::set<void*> nodes_in;
          for(size_t i=blk0_start;i<blk0_end;i++){
            nodes_out_.push_back((FMMNode*)nodes_out[i]);
            Vector<FMMNode*>& lst=((FMMNode*)nodes_out[i])->interac_list[interac_type];
            for(size_t k=0;k<mat_cnt;k++) if(lst[k]!=NULL && lst[k]->pt_cnt[0]) nodes_in.insert(lst[k]);
          }
          for(std::set<void*>::iterator node=nodes_in.begin(); node != nodes_in.end(); node++){
            nodes_in_.push_back((FMMNode*)*node);
          }

          size_t  input_dim=nodes_in_ .size()*ker_dim0*dof*fftsize;
          size_t output_dim=nodes_out_.size()*ker_dim1*dof*fftsize;
          size_t buffer_dim=2*(ker_dim0+ker_dim1)*dof*fftsize*omp_p;
          if(buff_size<(input_dim + output_dim + buffer_dim)*sizeof(Real_t))
            buff_size=(input_dim + output_dim + buffer_dim)*sizeof(Real_t);
        }

        { // Set fft vectors.
          for(size_t i=0;i<nodes_in_ .size();i++) fft_vec[blk0].push_back((size_t)(& input_vector[nodes_in_[i]->node_id][0][0]- input_data[0]));
          for(size_t i=0;i<nodes_out_.size();i++)ifft_vec[blk0].push_back((size_t)(&output_vector[blk0_start   +     i ][0][0]-output_data[0]));

          size_t scal_dim0=src_scal.Dim();
          size_t scal_dim1=trg_scal.Dim();
          fft_scl [blk0].resize(nodes_in_ .size()*scal_dim0);
          ifft_scl[blk0].resize(nodes_out_.size()*scal_dim1);
          for(size_t i=0;i<nodes_in_ .size();i++){
            size_t depth=nodes_in_[i]->Depth()+1;
            for(size_t j=0;j<scal_dim0;j++){
              fft_scl[blk0][i*scal_dim0+j]=pvfmm::pow<Real_t>(2.0, src_scal[j]*depth);
            }
          }
          for(size_t i=0;i<nodes_out_.size();i++){
            size_t depth=nodes_out_[i]->Depth()+1;
            for(size_t j=0;j<scal_dim1;j++){
              ifft_scl[blk0][i*scal_dim1+j]=pvfmm::pow<Real_t>(2.0, trg_scal[j]*depth);
            }
          }
        }
      }

      for(size_t blk0=0;blk0<n_blk0;blk0++){ // Hadamard interactions.
        std::vector<FMMNode*>& nodes_in_ =nodes_blk_in [blk0];
        std::vector<FMMNode*>& nodes_out_=nodes_blk_out[blk0];
        for(size_t i=0;i<nodes_in_.size();i++) nodes_in_[i]->node_id=i;
        { // Next blocking level.
          size_t n_blk1=nodes_out_.size()*(2)*sizeof(Real_t)/(64*V_BLK_CACHE);
          if(n_blk1==0) n_blk1=1;

          size_t interac_dsp_=0;
          for(size_t blk1=0;blk1<n_blk1;blk1++){
            size_t blk1_start=(nodes_out_.size()* blk1   )/n_blk1;
            size_t blk1_end  =(nodes_out_.size()*(blk1+1))/n_blk1;
            for(size_t k=0;k<mat_cnt;k++){
              for(size_t i=blk1_start;i<blk1_end;i++){
                Vector<FMMNode*>& lst=((FMMNode*)nodes_out_[i])->interac_list[interac_type];
                if(lst[k]!=NULL && lst[k]->pt_cnt[0]){
                  interac_vec[blk0].push_back(lst[k]->node_id*fftsize*ker_dim0*dof);
                  interac_vec[blk0].push_back(    i          *fftsize*ker_dim1*dof);
                  interac_dsp_++;
                }
              }
              interac_dsp[blk0].push_back(interac_dsp_);
            }
          }
        }
      }
    }

    { // Set interac_data.
      size_t data_size=sizeof(size_t)*6; // buff_size, m, dof, ker_dim0, ker_dim1, n_blk0
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        data_size+=sizeof(size_t)+    fft_vec[blk0].size()*sizeof(size_t);
        data_size+=sizeof(size_t)+   ifft_vec[blk0].size()*sizeof(size_t);
        data_size+=sizeof(size_t)+    fft_scl[blk0].size()*sizeof(Real_t);
        data_size+=sizeof(size_t)+   ifft_scl[blk0].size()*sizeof(Real_t);
        data_size+=sizeof(size_t)+interac_vec[blk0].size()*sizeof(size_t);
        data_size+=sizeof(size_t)+interac_dsp[blk0].size()*sizeof(size_t);
      }
      data_size+=sizeof(size_t)+interac_mat.size()*sizeof(size_t);
      data_size+=sizeof(size_t)+interac_mat_ptr.size()*sizeof(Real_t*);
      if(data_size>interac_data.Dim(0)*interac_data.Dim(1))
        interac_data.ReInit(1,data_size);
      char* data_ptr=&interac_data[0][0];

      ((size_t*)data_ptr)[0]=buff_size; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=        m; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=      dof; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]= ker_dim0; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]= ker_dim1; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=   n_blk0; data_ptr+=sizeof(size_t);

      ((size_t*)data_ptr)[0]= interac_mat.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &interac_mat[0], interac_mat.size()*sizeof(size_t));
      data_ptr+=interac_mat.size()*sizeof(size_t);

      ((size_t*)data_ptr)[0]= interac_mat_ptr.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &interac_mat_ptr[0], interac_mat_ptr.size()*sizeof(Real_t*));
      data_ptr+=interac_mat_ptr.size()*sizeof(Real_t*);

      for(size_t blk0=0;blk0<n_blk0;blk0++){
        ((size_t*)data_ptr)[0]= fft_vec[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, & fft_vec[blk0][0],  fft_vec[blk0].size()*sizeof(size_t));
        data_ptr+= fft_vec[blk0].size()*sizeof(size_t);

        ((size_t*)data_ptr)[0]=ifft_vec[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &ifft_vec[blk0][0], ifft_vec[blk0].size()*sizeof(size_t));
        data_ptr+=ifft_vec[blk0].size()*sizeof(size_t);

        ((size_t*)data_ptr)[0]= fft_scl[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, & fft_scl[blk0][0],  fft_scl[blk0].size()*sizeof(Real_t));
        data_ptr+= fft_scl[blk0].size()*sizeof(Real_t);

        ((size_t*)data_ptr)[0]=ifft_scl[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &ifft_scl[blk0][0], ifft_scl[blk0].size()*sizeof(Real_t));
        data_ptr+=ifft_scl[blk0].size()*sizeof(Real_t);

        ((size_t*)data_ptr)[0]=interac_vec[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_vec[blk0][0], interac_vec[blk0].size()*sizeof(size_t));
        data_ptr+=interac_vec[blk0].size()*sizeof(size_t);

        ((size_t*)data_ptr)[0]=interac_dsp[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_dsp[blk0][0], interac_dsp[blk0].size()*sizeof(size_t));
        data_ptr+=interac_dsp[blk0].size()*sizeof(size_t);
      }
    }
  }
  Profile::Toc();

  if(device){ // Host2Device
    Profile::Tic("Host2Device",&this->comm,false,25);
    setup_data.interac_data. AllocDevice(true);
    Profile::Toc();
  }
}

template <class FMMNode>
void FMM_Pts<FMMNode>::V_List     (SetupData<Real_t>&  setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  assert(!device); //Can not run on accelerator yet.

  int np;
  MPI_Comm_size(comm,&np);
  if(setup_data.interac_data.Dim(0)==0 || setup_data.interac_data.Dim(1)==0){
    if(np>1) Profile::Tic("Host2Device",&this->comm,false,25);
    if(np>1) Profile::Toc();
    return;
  }

  Profile::Tic("Host2Device",&this->comm,false,25);
  int level=setup_data.level;
  size_t buff_size=*((size_t*)&setup_data.interac_data[0][0]);
  typename Vector<char>::Device          buff;
  //typename Matrix<char>::Device  precomp_data;
  typename Matrix<char>::Device  interac_data;
  typename Matrix<Real_t>::Device  input_data;
  typename Matrix<Real_t>::Device output_data;

  if(device){
    if(this->dev_buffer.Dim()<buff_size) this->dev_buffer.ReInit(buff_size);
    buff        =       this-> dev_buffer. AllocDevice(false);
    //precomp_data= setup_data.precomp_data->AllocDevice(false);
    interac_data= setup_data.interac_data. AllocDevice(false);
    input_data  = setup_data.  input_data->AllocDevice(false);
    output_data = setup_data. output_data->AllocDevice(false);
  }else{
    if(this->dev_buffer.Dim()<buff_size) this->dev_buffer.ReInit(buff_size);
    buff        =       this-> dev_buffer;
    //precomp_data=*setup_data.precomp_data;
    interac_data= setup_data.interac_data;
    input_data  =*setup_data.  input_data;
    output_data =*setup_data. output_data;
  }
  Profile::Toc();

  { // Offloaded computation.

    // Set interac_data.
    size_t m, dof, ker_dim0, ker_dim1, n_blk0;
    std::vector<Vector<size_t> >  fft_vec;
    std::vector<Vector<size_t> > ifft_vec;
    std::vector<Vector<Real_t> >  fft_scl;
    std::vector<Vector<Real_t> > ifft_scl;
    std::vector<Vector<size_t> > interac_vec;
    std::vector<Vector<size_t> > interac_dsp;
    Vector<Real_t*> precomp_mat;
    { // Set interac_data.
      char* data_ptr=&interac_data[0][0];

      buff_size=((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      m        =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      dof      =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      ker_dim0 =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      ker_dim1 =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      n_blk0   =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);

      fft_vec .resize(n_blk0);
      ifft_vec.resize(n_blk0);
      fft_scl .resize(n_blk0);
      ifft_scl.resize(n_blk0);
      interac_vec.resize(n_blk0);
      interac_dsp.resize(n_blk0);

      Vector<size_t> interac_mat;
      interac_mat.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+interac_mat.Dim()*sizeof(size_t);

      Vector<Real_t*> interac_mat_ptr;
      interac_mat_ptr.ReInit(((size_t*)data_ptr)[0],(Real_t**)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+interac_mat_ptr.Dim()*sizeof(Real_t*);

#if 0 // Since we skip SetupPrecomp for V-list
      precomp_mat.Resize(interac_mat.Dim());
      for(size_t i=0;i<interac_mat.Dim();i++){
        precomp_mat[i]=(Real_t*)(precomp_data[0]+interac_mat[i]);
      }
#else
      precomp_mat.Resize(interac_mat_ptr.Dim());
      for(size_t i=0;i<interac_mat_ptr.Dim();i++){
        precomp_mat[i]=interac_mat_ptr[i];
      }
#endif

      for(size_t blk0=0;blk0<n_blk0;blk0++){
        fft_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+fft_vec[blk0].Dim()*sizeof(size_t);

        ifft_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+ifft_vec[blk0].Dim()*sizeof(size_t);

        fft_scl[blk0].ReInit(((size_t*)data_ptr)[0],(Real_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+fft_scl[blk0].Dim()*sizeof(Real_t);

        ifft_scl[blk0].ReInit(((size_t*)data_ptr)[0],(Real_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+ifft_scl[blk0].Dim()*sizeof(Real_t);

        interac_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+interac_vec[blk0].Dim()*sizeof(size_t);

        interac_dsp[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+interac_dsp[blk0].Dim()*sizeof(size_t);
      }
    }

    int omp_p=omp_get_max_threads();
    size_t M_dim, fftsize;
    {
      size_t n1=m*2;
      size_t n2=n1*n1;
      size_t n3_=n2*(n1/2+1);
      size_t chld_cnt=1UL<<COORD_DIM;
      fftsize=2*n3_*chld_cnt;
      M_dim=n3_;
    }

    for(size_t blk0=0;blk0<n_blk0;blk0++){ // interactions
      size_t n_in = fft_vec[blk0].Dim();
      size_t n_out=ifft_vec[blk0].Dim();

      size_t  input_dim=n_in *ker_dim0*dof*fftsize;
      size_t output_dim=n_out*ker_dim1*dof*fftsize;
      size_t buffer_dim=2*(ker_dim0+ker_dim1)*dof*fftsize*omp_p;

      Vector<Real_t> fft_in ( input_dim, (Real_t*)&buff[         0                           ],false);
      Vector<Real_t> fft_out(output_dim, (Real_t*)&buff[ input_dim            *sizeof(Real_t)],false);
      Vector<Real_t>  buffer(buffer_dim, (Real_t*)&buff[(input_dim+output_dim)*sizeof(Real_t)],false);

      { //  FFT
        if(np==1) Profile::Tic("FFT",&comm,false,100);
        Vector<Real_t>  input_data_( input_data.dim[0]* input_data.dim[1],  input_data[0], false);
        FFT_UpEquiv(dof, m, ker_dim0,  fft_vec[blk0],  fft_scl[blk0],  input_data_, fft_in, buffer);
        if(np==1) Profile::Toc();
      }
      { // Hadamard
#ifdef PVFMM_HAVE_PAPI
#ifdef __VERBOSE__
        std::cout << "Starting counters new\n";
        if (PAPI_start(EventSet) != PAPI_OK) std::cout << "handle_error3" << std::endl;
#endif
#endif
        if(np==1) Profile::Tic("HadamardProduct",&comm,false,100);
        VListHadamard<Real_t>(dof, M_dim, ker_dim0, ker_dim1, interac_dsp[blk0], interac_vec[blk0], precomp_mat, fft_in, fft_out);
        if(np==1) Profile::Toc();
#ifdef PVFMM_HAVE_PAPI
#ifdef __VERBOSE__
        if (PAPI_stop(EventSet, values) != PAPI_OK) std::cout << "handle_error4" << std::endl;
        std::cout << "Stopping counters\n";
#endif
#endif
      }
      { // IFFT
        if(np==1) Profile::Tic("IFFT",&comm,false,100);
        Vector<Real_t> output_data_(output_data.dim[0]*output_data.dim[1], output_data[0], false);
        FFT_Check2Equiv(dof, m, ker_dim1, ifft_vec[blk0], ifft_scl[blk0], fft_out, output_data_, buffer);
        if(np==1) Profile::Toc();
      }
    }
  }
}


template <class FMMNode>
void FMM_Pts<FMMNode>::Down2DownSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=kernel->k_l2l;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=D2D_Type;

    setup_data. input_data=&buff[1];
    setup_data.output_data=&buff[1];
    Vector<FMMNode_t*>& nodes_in =n_list[1];
    Vector<FMMNode_t*>& nodes_out=n_list[1];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->Depth()==level-1) && nodes_in [i]->pt_cnt[1]) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->Depth()==level  ) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode*)nodes_in [i])->FMMData())->dnward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)nodes_out[i])->FMMData())->dnward_equiv);

  SetupInterac(setup_data,device);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::Down2Down     (SetupData<Real_t>& setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add Down2Down contribution.
  EvalList(setup_data, device);
}


template <class FMMNode>
void FMM_Pts<FMMNode>::PtSetup(SetupData<Real_t>& setup_data, void* data_){
  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };
  ptSetupData& data=*(ptSetupData*)data_;
  if(data.interac_data.interac_cnt.Dim()){ // Set data.interac_data.interac_cst
    InteracData& intdata=data.interac_data;

    Vector<size_t>  cnt;
    Vector<size_t>& dsp=intdata.interac_cst;
    cnt.ReInit(intdata.interac_cnt.Dim());
    dsp.ReInit(intdata.interac_dsp.Dim());
    #pragma omp parallel for
    for(size_t trg=0;trg<cnt.Dim();trg++){
      size_t trg_cnt=data.trg_coord.cnt[trg];

      cnt[trg]=0;
      for(size_t i=0;i<intdata.interac_cnt[trg];i++){
        size_t int_id=intdata.interac_dsp[trg]+i;
        size_t src=intdata.in_node[int_id];

        size_t src_cnt=data.src_coord.cnt[src];
        size_t srf_cnt=data.srf_coord.cnt[src];
        cnt[trg]+=(src_cnt+srf_cnt)*trg_cnt;
      }
    }

    dsp[0]=cnt[0];
    omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
  }
  { // pack data
    struct PackedSetupData{
      size_t size;

      int level;
      const Kernel<Real_t>* kernel;

      Matrix<Real_t>* src_coord; // Src coord
      Matrix<Real_t>* src_value; // Src density
      Matrix<Real_t>* srf_coord; // Srf coord
      Matrix<Real_t>* srf_value; // Srf density
      Matrix<Real_t>* trg_coord; // Trg coord
      Matrix<Real_t>* trg_value; // Trg potential

      size_t src_coord_cnt_size; size_t src_coord_cnt_offset;
      size_t src_coord_dsp_size; size_t src_coord_dsp_offset;
      size_t src_value_cnt_size; size_t src_value_cnt_offset;
      size_t src_value_dsp_size; size_t src_value_dsp_offset;

      size_t srf_coord_cnt_size; size_t srf_coord_cnt_offset;
      size_t srf_coord_dsp_size; size_t srf_coord_dsp_offset;
      size_t srf_value_cnt_size; size_t srf_value_cnt_offset;
      size_t srf_value_dsp_size; size_t srf_value_dsp_offset;

      size_t trg_coord_cnt_size; size_t trg_coord_cnt_offset;
      size_t trg_coord_dsp_size; size_t trg_coord_dsp_offset;
      size_t trg_value_cnt_size; size_t trg_value_cnt_offset;
      size_t trg_value_dsp_size; size_t trg_value_dsp_offset;

      // interac_data
      size_t          in_node_size; size_t           in_node_offset;
      size_t         scal_idx_size; size_t          scal_idx_offset;
      size_t      coord_shift_size; size_t       coord_shift_offset;
      size_t      interac_cnt_size; size_t       interac_cnt_offset;
      size_t      interac_dsp_size; size_t       interac_dsp_offset;
      size_t      interac_cst_size; size_t       interac_cst_offset;
      size_t scal_dim[4*MAX_DEPTH]; size_t scal_offset[4*MAX_DEPTH];
      size_t            Mdim[4][2]; size_t              M_offset[4];
    };
    PackedSetupData pkd_data;
    { // Set pkd_data
      size_t offset=mem::align_ptr(sizeof(PackedSetupData));

      pkd_data. level=data. level;
      pkd_data.kernel=data.kernel;

      pkd_data.src_coord=data.src_coord.ptr;
      pkd_data.src_value=data.src_value.ptr;
      pkd_data.srf_coord=data.srf_coord.ptr;
      pkd_data.srf_value=data.srf_value.ptr;
      pkd_data.trg_coord=data.trg_coord.ptr;
      pkd_data.trg_value=data.trg_value.ptr;

      pkd_data.src_coord_cnt_offset=offset; pkd_data.src_coord_cnt_size=data.src_coord.cnt.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.src_coord_cnt_size);
      pkd_data.src_coord_dsp_offset=offset; pkd_data.src_coord_dsp_size=data.src_coord.dsp.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.src_coord_dsp_size);
      pkd_data.src_value_cnt_offset=offset; pkd_data.src_value_cnt_size=data.src_value.cnt.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.src_value_cnt_size);
      pkd_data.src_value_dsp_offset=offset; pkd_data.src_value_dsp_size=data.src_value.dsp.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.src_value_dsp_size);

      pkd_data.srf_coord_cnt_offset=offset; pkd_data.srf_coord_cnt_size=data.srf_coord.cnt.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.srf_coord_cnt_size);
      pkd_data.srf_coord_dsp_offset=offset; pkd_data.srf_coord_dsp_size=data.srf_coord.dsp.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.srf_coord_dsp_size);
      pkd_data.srf_value_cnt_offset=offset; pkd_data.srf_value_cnt_size=data.srf_value.cnt.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.srf_value_cnt_size);
      pkd_data.srf_value_dsp_offset=offset; pkd_data.srf_value_dsp_size=data.srf_value.dsp.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.srf_value_dsp_size);

      pkd_data.trg_coord_cnt_offset=offset; pkd_data.trg_coord_cnt_size=data.trg_coord.cnt.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.trg_coord_cnt_size);
      pkd_data.trg_coord_dsp_offset=offset; pkd_data.trg_coord_dsp_size=data.trg_coord.dsp.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.trg_coord_dsp_size);
      pkd_data.trg_value_cnt_offset=offset; pkd_data.trg_value_cnt_size=data.trg_value.cnt.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.trg_value_cnt_size);
      pkd_data.trg_value_dsp_offset=offset; pkd_data.trg_value_dsp_size=data.trg_value.dsp.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.trg_value_dsp_size);


      InteracData& intdata=data.interac_data;
      pkd_data.    in_node_offset=offset; pkd_data.    in_node_size=intdata.    in_node.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.    in_node_size);
      pkd_data.   scal_idx_offset=offset; pkd_data.   scal_idx_size=intdata.   scal_idx.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.   scal_idx_size);
      pkd_data.coord_shift_offset=offset; pkd_data.coord_shift_size=intdata.coord_shift.Dim(); offset+=mem::align_ptr(sizeof(Real_t)*pkd_data.coord_shift_size);
      pkd_data.interac_cnt_offset=offset; pkd_data.interac_cnt_size=intdata.interac_cnt.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.interac_cnt_size);
      pkd_data.interac_dsp_offset=offset; pkd_data.interac_dsp_size=intdata.interac_dsp.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.interac_dsp_size);
      pkd_data.interac_cst_offset=offset; pkd_data.interac_cst_size=intdata.interac_cst.Dim(); offset+=mem::align_ptr(sizeof(size_t)*pkd_data.interac_cst_size);

      for(size_t i=0;i<4*MAX_DEPTH;i++){
        pkd_data.scal_offset[i]=offset; pkd_data.scal_dim[i]=intdata.scal[i].Dim(); offset+=mem::align_ptr(sizeof(Real_t)*pkd_data.scal_dim[i]);
      }
      for(size_t i=0;i<4;i++){
        size_t& Mdim0=pkd_data.Mdim[i][0];
        size_t& Mdim1=pkd_data.Mdim[i][1];
        pkd_data.M_offset[i]=offset; Mdim0=intdata.M[i].Dim(0); Mdim1=intdata.M[i].Dim(1); offset+=mem::align_ptr(sizeof(Real_t)*Mdim0*Mdim1);
      }

      pkd_data.size=offset;
    }
    { // Set setup_data.interac_data
      Matrix<char>& buff=setup_data.interac_data;
      if(pkd_data.size>buff.Dim(0)*buff.Dim(1)){
        buff.ReInit(1,pkd_data.size);
      }
      ((PackedSetupData*)buff[0])[0]=pkd_data;

      if(pkd_data.src_coord_cnt_size) memcpy(&buff[0][pkd_data.src_coord_cnt_offset], &data.src_coord.cnt[0], pkd_data.src_coord_cnt_size*sizeof(size_t));
      if(pkd_data.src_coord_dsp_size) memcpy(&buff[0][pkd_data.src_coord_dsp_offset], &data.src_coord.dsp[0], pkd_data.src_coord_dsp_size*sizeof(size_t));
      if(pkd_data.src_value_cnt_size) memcpy(&buff[0][pkd_data.src_value_cnt_offset], &data.src_value.cnt[0], pkd_data.src_value_cnt_size*sizeof(size_t));
      if(pkd_data.src_value_dsp_size) memcpy(&buff[0][pkd_data.src_value_dsp_offset], &data.src_value.dsp[0], pkd_data.src_value_dsp_size*sizeof(size_t));

      if(pkd_data.srf_coord_cnt_size) memcpy(&buff[0][pkd_data.srf_coord_cnt_offset], &data.srf_coord.cnt[0], pkd_data.srf_coord_cnt_size*sizeof(size_t));
      if(pkd_data.srf_coord_dsp_size) memcpy(&buff[0][pkd_data.srf_coord_dsp_offset], &data.srf_coord.dsp[0], pkd_data.srf_coord_dsp_size*sizeof(size_t));
      if(pkd_data.srf_value_cnt_size) memcpy(&buff[0][pkd_data.srf_value_cnt_offset], &data.srf_value.cnt[0], pkd_data.srf_value_cnt_size*sizeof(size_t));
      if(pkd_data.srf_value_dsp_size) memcpy(&buff[0][pkd_data.srf_value_dsp_offset], &data.srf_value.dsp[0], pkd_data.srf_value_dsp_size*sizeof(size_t));

      if(pkd_data.trg_coord_cnt_size) memcpy(&buff[0][pkd_data.trg_coord_cnt_offset], &data.trg_coord.cnt[0], pkd_data.trg_coord_cnt_size*sizeof(size_t));
      if(pkd_data.trg_coord_dsp_size) memcpy(&buff[0][pkd_data.trg_coord_dsp_offset], &data.trg_coord.dsp[0], pkd_data.trg_coord_dsp_size*sizeof(size_t));
      if(pkd_data.trg_value_cnt_size) memcpy(&buff[0][pkd_data.trg_value_cnt_offset], &data.trg_value.cnt[0], pkd_data.trg_value_cnt_size*sizeof(size_t));
      if(pkd_data.trg_value_dsp_size) memcpy(&buff[0][pkd_data.trg_value_dsp_offset], &data.trg_value.dsp[0], pkd_data.trg_value_dsp_size*sizeof(size_t));


      InteracData& intdata=data.interac_data;
      if(pkd_data.    in_node_size) memcpy(&buff[0][pkd_data.    in_node_offset], &intdata.    in_node[0], pkd_data.    in_node_size*sizeof(size_t));
      if(pkd_data.   scal_idx_size) memcpy(&buff[0][pkd_data.   scal_idx_offset], &intdata.   scal_idx[0], pkd_data.   scal_idx_size*sizeof(size_t));
      if(pkd_data.coord_shift_size) memcpy(&buff[0][pkd_data.coord_shift_offset], &intdata.coord_shift[0], pkd_data.coord_shift_size*sizeof(Real_t));
      if(pkd_data.interac_cnt_size) memcpy(&buff[0][pkd_data.interac_cnt_offset], &intdata.interac_cnt[0], pkd_data.interac_cnt_size*sizeof(size_t));
      if(pkd_data.interac_dsp_size) memcpy(&buff[0][pkd_data.interac_dsp_offset], &intdata.interac_dsp[0], pkd_data.interac_dsp_size*sizeof(size_t));
      if(pkd_data.interac_cst_size) memcpy(&buff[0][pkd_data.interac_cst_offset], &intdata.interac_cst[0], pkd_data.interac_cst_size*sizeof(size_t));
      for(size_t i=0;i<4*MAX_DEPTH;i++){
        if(intdata.scal[i].Dim()) memcpy(&buff[0][pkd_data.scal_offset[i]], &intdata.scal[i][0], intdata.scal[i].Dim()*sizeof(Real_t));
      }
      for(size_t i=0;i<4;i++){
        if(intdata.M[i].Dim(0)*intdata.M[i].Dim(1)) memcpy(&buff[0][pkd_data.M_offset[i]], &intdata.M[i][0][0], intdata.M[i].Dim(0)*intdata.M[i].Dim(1)*sizeof(Real_t));
      }
    }
  }
  { // Resize device buffer
    size_t n=setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1)*sizeof(Real_t);
    if(this->dev_buffer.Dim()<n) this->dev_buffer.ReInit(n);
  }
}

template <class FMMNode>
template <int SYNC>
void FMM_Pts<FMMNode>::EvalListPts(SetupData<Real_t>& setup_data, bool device){
  if(setup_data.kernel->ker_dim[0]*setup_data.kernel->ker_dim[1]==0) return;
  if(setup_data.interac_data.Dim(0)==0 || setup_data.interac_data.Dim(1)==0){
    Profile::Tic("Host2Device",&this->comm,false,25);
    Profile::Toc();
    Profile::Tic("DeviceComp",&this->comm,false,20);
    Profile::Toc();
    return;
  }

  bool have_gpu=false;
  #if defined(PVFMM_HAVE_CUDA)
  have_gpu=true;
  #endif

  Profile::Tic("Host2Device",&this->comm,false,25);
  typename Vector<char>::Device      dev_buff;
  typename Matrix<char>::Device  interac_data;
  typename Matrix<Real_t>::Device  coord_data;
  typename Matrix<Real_t>::Device  input_data;
  typename Matrix<Real_t>::Device output_data;
  size_t ptr_single_layer_kernel=(size_t)NULL;
  size_t ptr_double_layer_kernel=(size_t)NULL;
  if(device && !have_gpu){
    dev_buff    =       this-> dev_buffer. AllocDevice(false);
    interac_data= setup_data.interac_data. AllocDevice(false);
    if(setup_data.  coord_data!=NULL) coord_data  = setup_data.  coord_data->AllocDevice(false);
    if(setup_data.  input_data!=NULL) input_data  = setup_data.  input_data->AllocDevice(false);
    if(setup_data. output_data!=NULL) output_data = setup_data. output_data->AllocDevice(false);
    ptr_single_layer_kernel=setup_data.kernel->dev_ker_poten;
    ptr_double_layer_kernel=setup_data.kernel->dev_dbl_layer_poten;
  }else{
    dev_buff    =       this-> dev_buffer;
    interac_data= setup_data.interac_data;
    if(setup_data.  coord_data!=NULL) coord_data  =*setup_data.  coord_data;
    if(setup_data.  input_data!=NULL) input_data  =*setup_data.  input_data;
    if(setup_data. output_data!=NULL) output_data =*setup_data. output_data;
    ptr_single_layer_kernel=(size_t)setup_data.kernel->ker_poten;
    ptr_double_layer_kernel=(size_t)setup_data.kernel->dbl_layer_poten;
  }
  Profile::Toc();

  Profile::Tic("DeviceComp",&this->comm,false,20);
  int lock_idx=-1;
  int wait_lock_idx=-1;
  if(device) wait_lock_idx=MIC_Lock::curr_lock();
  if(device) lock_idx=MIC_Lock::get_lock();
  #ifdef __INTEL_OFFLOAD
  #pragma offload if(device) target(mic:0) signal(&MIC_Lock::lock_vec[device?lock_idx:0])
  #endif
  { // Offloaded computation.
    struct PackedData{
      size_t len;
      Matrix<Real_t>* ptr;
      Vector<size_t> cnt;
      Vector<size_t> dsp;
    };
    struct InteracData{
      Vector<size_t> in_node;
      Vector<size_t> scal_idx;
      Vector<Real_t> coord_shift;
      Vector<size_t> interac_cnt;
      Vector<size_t> interac_dsp;
      Vector<size_t> interac_cst;
      Vector<Real_t> scal[4*MAX_DEPTH];
      Matrix<Real_t> M[4];
    };
    struct ptSetupData{
      int level;
      const Kernel<Real_t>* kernel;

      PackedData src_coord; // Src coord
      PackedData src_value; // Src density
      PackedData srf_coord; // Srf coord
      PackedData srf_value; // Srf density
      PackedData trg_coord; // Trg coord
      PackedData trg_value; // Trg potential

      InteracData interac_data;
    };

    ptSetupData data;
    { // Initialize data
      struct PackedSetupData{
        size_t size;

        int level;
        const Kernel<Real_t>* kernel;

        Matrix<Real_t>* src_coord; // Src coord
        Matrix<Real_t>* src_value; // Src density
        Matrix<Real_t>* srf_coord; // Srf coord
        Matrix<Real_t>* srf_value; // Srf density
        Matrix<Real_t>* trg_coord; // Trg coord
        Matrix<Real_t>* trg_value; // Trg potential

        size_t src_coord_cnt_size; size_t src_coord_cnt_offset;
        size_t src_coord_dsp_size; size_t src_coord_dsp_offset;
        size_t src_value_cnt_size; size_t src_value_cnt_offset;
        size_t src_value_dsp_size; size_t src_value_dsp_offset;

        size_t srf_coord_cnt_size; size_t srf_coord_cnt_offset;
        size_t srf_coord_dsp_size; size_t srf_coord_dsp_offset;
        size_t srf_value_cnt_size; size_t srf_value_cnt_offset;
        size_t srf_value_dsp_size; size_t srf_value_dsp_offset;

        size_t trg_coord_cnt_size; size_t trg_coord_cnt_offset;
        size_t trg_coord_dsp_size; size_t trg_coord_dsp_offset;
        size_t trg_value_cnt_size; size_t trg_value_cnt_offset;
        size_t trg_value_dsp_size; size_t trg_value_dsp_offset;

        // interac_data
        size_t          in_node_size; size_t           in_node_offset;
        size_t         scal_idx_size; size_t          scal_idx_offset;
        size_t      coord_shift_size; size_t       coord_shift_offset;
        size_t      interac_cnt_size; size_t       interac_cnt_offset;
        size_t      interac_dsp_size; size_t       interac_dsp_offset;
        size_t      interac_cst_size; size_t       interac_cst_offset;
        size_t scal_dim[4*MAX_DEPTH]; size_t scal_offset[4*MAX_DEPTH];
        size_t            Mdim[4][2]; size_t              M_offset[4];
      };
      typename Matrix<char>::Device& setupdata=interac_data;
      PackedSetupData& pkd_data=*((PackedSetupData*)setupdata[0]);

      data. level=pkd_data. level;
      data.kernel=pkd_data.kernel;

      data.src_coord.ptr=pkd_data.src_coord;
      data.src_value.ptr=pkd_data.src_value;
      data.srf_coord.ptr=pkd_data.srf_coord;
      data.srf_value.ptr=pkd_data.srf_value;
      data.trg_coord.ptr=pkd_data.trg_coord;
      data.trg_value.ptr=pkd_data.trg_value;


      data.src_coord.cnt.ReInit(pkd_data.src_coord_cnt_size, (size_t*)&setupdata[0][pkd_data.src_coord_cnt_offset], false);
      data.src_coord.dsp.ReInit(pkd_data.src_coord_dsp_size, (size_t*)&setupdata[0][pkd_data.src_coord_dsp_offset], false);
      data.src_value.cnt.ReInit(pkd_data.src_value_cnt_size, (size_t*)&setupdata[0][pkd_data.src_value_cnt_offset], false);
      data.src_value.dsp.ReInit(pkd_data.src_value_dsp_size, (size_t*)&setupdata[0][pkd_data.src_value_dsp_offset], false);

      data.srf_coord.cnt.ReInit(pkd_data.srf_coord_cnt_size, (size_t*)&setupdata[0][pkd_data.srf_coord_cnt_offset], false);
      data.srf_coord.dsp.ReInit(pkd_data.srf_coord_dsp_size, (size_t*)&setupdata[0][pkd_data.srf_coord_dsp_offset], false);
      data.srf_value.cnt.ReInit(pkd_data.srf_value_cnt_size, (size_t*)&setupdata[0][pkd_data.srf_value_cnt_offset], false);
      data.srf_value.dsp.ReInit(pkd_data.srf_value_dsp_size, (size_t*)&setupdata[0][pkd_data.srf_value_dsp_offset], false);

      data.trg_coord.cnt.ReInit(pkd_data.trg_coord_cnt_size, (size_t*)&setupdata[0][pkd_data.trg_coord_cnt_offset], false);
      data.trg_coord.dsp.ReInit(pkd_data.trg_coord_dsp_size, (size_t*)&setupdata[0][pkd_data.trg_coord_dsp_offset], false);
      data.trg_value.cnt.ReInit(pkd_data.trg_value_cnt_size, (size_t*)&setupdata[0][pkd_data.trg_value_cnt_offset], false);
      data.trg_value.dsp.ReInit(pkd_data.trg_value_dsp_size, (size_t*)&setupdata[0][pkd_data.trg_value_dsp_offset], false);


      InteracData& intdata=data.interac_data;
      intdata.    in_node.ReInit(pkd_data.    in_node_size, (size_t*)&setupdata[0][pkd_data.    in_node_offset],false);
      intdata.   scal_idx.ReInit(pkd_data.   scal_idx_size, (size_t*)&setupdata[0][pkd_data.   scal_idx_offset],false);
      intdata.coord_shift.ReInit(pkd_data.coord_shift_size, (Real_t*)&setupdata[0][pkd_data.coord_shift_offset],false);
      intdata.interac_cnt.ReInit(pkd_data.interac_cnt_size, (size_t*)&setupdata[0][pkd_data.interac_cnt_offset],false);
      intdata.interac_dsp.ReInit(pkd_data.interac_dsp_size, (size_t*)&setupdata[0][pkd_data.interac_dsp_offset],false);
      intdata.interac_cst.ReInit(pkd_data.interac_cst_size, (size_t*)&setupdata[0][pkd_data.interac_cst_offset],false);
      for(size_t i=0;i<4*MAX_DEPTH;i++){
        intdata.scal[i].ReInit(pkd_data.scal_dim[i], (Real_t*)&setupdata[0][pkd_data.scal_offset[i]],false);
      }
      for(size_t i=0;i<4;i++){
        intdata.M[i].ReInit(pkd_data.Mdim[i][0], pkd_data.Mdim[i][1], (Real_t*)&setupdata[0][pkd_data.M_offset[i]],false);
      }
    }

    if(device) MIC_Lock::wait_lock(wait_lock_idx);
    { // Compute interactions
      InteracData& intdata=data.interac_data;
      typename Kernel<Real_t>::Ker_t single_layer_kernel=(typename Kernel<Real_t>::Ker_t)ptr_single_layer_kernel;
      typename Kernel<Real_t>::Ker_t double_layer_kernel=(typename Kernel<Real_t>::Ker_t)ptr_double_layer_kernel;
      int omp_p=omp_get_max_threads();

      #pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){

        Matrix<Real_t> src_coord, src_value;
        Matrix<Real_t> srf_coord, srf_value;
        Matrix<Real_t> trg_coord, trg_value;
        Vector<Real_t> buff;
        { // init buff
          size_t thread_buff_size=dev_buff.dim/sizeof(Real_t)/omp_p;
          buff.ReInit(thread_buff_size, (Real_t*)&dev_buff[tid*thread_buff_size*sizeof(Real_t)], false);
        }

        size_t vcnt=0;
        std::vector<Matrix<Real_t> > vbuff(6);
        { // init vbuff[0:5]
          size_t vdim_=0, vdim[6];
          for(size_t indx=0;indx<6;indx++){
            vdim[indx]=0;
            switch(indx){
              case 0:
                vdim[indx]=intdata.M[0].Dim(0); break;
              case 1:
                assert(intdata.M[0].Dim(1)==intdata.M[1].Dim(0));
                vdim[indx]=intdata.M[0].Dim(1); break;
              case 2:
                vdim[indx]=intdata.M[1].Dim(1); break;
              case 3:
                vdim[indx]=intdata.M[2].Dim(0); break;
              case 4:
                assert(intdata.M[2].Dim(1)==intdata.M[3].Dim(0));
                vdim[indx]=intdata.M[2].Dim(1); break;
              case 5:
                vdim[indx]=intdata.M[3].Dim(1); break;
              default:
                vdim[indx]=0; break;
            }
            vdim_+=vdim[indx];
          }
          if(vdim_){
            vcnt=buff.Dim()/vdim_/2;
            assert(vcnt>0); // Thread buffer is too small
          }

          for(size_t indx=0;indx<6;indx++){ // init vbuff[0:5]
            vbuff[indx].ReInit(vcnt,vdim[indx],&buff[0],false);
            buff.ReInit(buff.Dim()-vdim[indx]*vcnt, &buff[vdim[indx]*vcnt], false);
          }
        }

        size_t trg_a=0, trg_b=0;
        if(intdata.interac_cst.Dim()){ // Determine trg_a, trg_b
          //trg_a=((tid+0)*intdata.interac_cnt.Dim())/omp_p;
          //trg_b=((tid+1)*intdata.interac_cnt.Dim())/omp_p;
          Vector<size_t>& interac_cst=intdata.interac_cst;
          size_t cost=interac_cst[interac_cst.Dim()-1];
          trg_a=std::lower_bound(&interac_cst[0],&interac_cst[interac_cst.Dim()-1],(cost*(tid+0))/omp_p)-&interac_cst[0]+1;
          trg_b=std::lower_bound(&interac_cst[0],&interac_cst[interac_cst.Dim()-1],(cost*(tid+1))/omp_p)-&interac_cst[0]+1;
          if(tid==omp_p-1) trg_b=interac_cst.Dim();
          if(tid==0) trg_a=0;
        }
        for(size_t trg0=trg_a;trg0<trg_b;){
          size_t trg1_max=1;
          if(vcnt){ // Find trg1_max
            size_t interac_cnt=intdata.interac_cnt[trg0];
            while(trg0+trg1_max<trg_b){
              interac_cnt+=intdata.interac_cnt[trg0+trg1_max];
              if(interac_cnt>vcnt){
                interac_cnt-=intdata.interac_cnt[trg0+trg1_max];
                break;
              }
              trg1_max++;
            }
            assert(interac_cnt<=vcnt);
            for(size_t k=0;k<6;k++){
              if(vbuff[k].Dim(0)*vbuff[k].Dim(1)){
                vbuff[k].ReInit(interac_cnt,vbuff[k].Dim(1),vbuff[k][0],false);
              }
            }
          }else{
            trg1_max=trg_b-trg0;
          }

          if(intdata.M[0].Dim(0) && intdata.M[0].Dim(1) && intdata.M[1].Dim(0) && intdata.M[1].Dim(1)){ // src mat-vec
            size_t interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){ // Copy src_value to vbuff[0]
              size_t trg=trg0+trg1;
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t src=intdata.in_node[int_id];
                src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
                { // Copy src_value to vbuff[0]
                  size_t vdim=vbuff[0].Dim(1);
                  assert(src_value.Dim(1)==vdim);
                  for(size_t j=0;j<vdim;j++) vbuff[0][interac_idx][j]=src_value[0][j];
                }
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[0];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+0];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                interac_idx++;
              }
            }

            Matrix<Real_t>::GEMM(vbuff[1],vbuff[0],intdata.M[0]);
            Matrix<Real_t>::GEMM(vbuff[2],vbuff[1],intdata.M[1]);

            interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[2];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+1];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                interac_idx++;
              }
            }
          }

          if(intdata.M[2].Dim(0) && intdata.M[2].Dim(1) && intdata.M[3].Dim(0) && intdata.M[3].Dim(1)){ // init vbuff[3]
            size_t vdim=vbuff[3].Dim(0)*vbuff[3].Dim(1);
            for(size_t i=0;i<vdim;i++) vbuff[3][0][i]=0;
          }

          { // Evaluate kernel functions
            size_t interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              trg_coord.ReInit(1, data.trg_coord.cnt[trg], &data.trg_coord.ptr[0][0][data.trg_coord.dsp[trg]], false);
              trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t src=intdata.in_node[int_id];
                src_coord.ReInit(1, data.src_coord.cnt[src], &data.src_coord.ptr[0][0][data.src_coord.dsp[src]], false);
                src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
                srf_coord.ReInit(1, data.srf_coord.cnt[src], &data.srf_coord.ptr[0][0][data.srf_coord.dsp[src]], false);
                srf_value.ReInit(1, data.srf_value.cnt[src], &data.srf_value.ptr[0][0][data.srf_value.dsp[src]], false);

                Real_t* vbuff2_ptr=(vbuff[2].Dim(0)*vbuff[2].Dim(1)?vbuff[2][interac_idx]:src_value[0]);
                Real_t* vbuff3_ptr=(vbuff[3].Dim(0)*vbuff[3].Dim(1)?vbuff[3][interac_idx]:trg_value[0]);

                if(src_coord.Dim(1)){
                  { // coord_shift
                    Real_t* shift=&intdata.coord_shift[int_id*COORD_DIM];
                    if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0){
                      size_t vdim=src_coord.Dim(1);
                      Vector<Real_t> new_coord(vdim, &buff[0], false);
                      assert(buff.Dim()>=vdim); // Thread buffer is too small
                      //buff.ReInit(buff.Dim()-vdim, &buff[vdim], false);
                      for(size_t j=0;j<vdim;j+=COORD_DIM){
                        for(size_t k=0;k<COORD_DIM;k++){
                          new_coord[j+k]=src_coord[0][j+k]+shift[k];
                        }
                      }
                      src_coord.ReInit(1, vdim, &new_coord[0], false);
                    }
                  }
                  assert(ptr_single_layer_kernel); // assert(Single-layer kernel is implemented)
                  single_layer_kernel(src_coord[0], src_coord.Dim(1)/COORD_DIM, vbuff2_ptr, 1,
                                      trg_coord[0], trg_coord.Dim(1)/COORD_DIM, vbuff3_ptr, NULL);
                }
                if(srf_coord.Dim(1)){
                  { // coord_shift
                    Real_t* shift=&intdata.coord_shift[int_id*COORD_DIM];
                    if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0){
                      size_t vdim=srf_coord.Dim(1);
                      Vector<Real_t> new_coord(vdim, &buff[0], false);
                      assert(buff.Dim()>=vdim); // Thread buffer is too small
                      //buff.ReInit(buff.Dim()-vdim, &buff[vdim], false);
                      for(size_t j=0;j<vdim;j+=COORD_DIM){
                        for(size_t k=0;k<COORD_DIM;k++){
                          new_coord[j+k]=srf_coord[0][j+k]+shift[k];
                        }
                      }
                      srf_coord.ReInit(1, vdim, &new_coord[0], false);
                    }
                  }
                  assert(ptr_double_layer_kernel); // assert(Double-layer kernel is implemented)
                  double_layer_kernel(srf_coord[0], srf_coord.Dim(1)/COORD_DIM, srf_value[0], 1,
                                      trg_coord[0], trg_coord.Dim(1)/COORD_DIM, vbuff3_ptr, NULL);
                }
                interac_idx++;
              }
            }
          }

          if(intdata.M[2].Dim(0) && intdata.M[2].Dim(1) && intdata.M[3].Dim(0) && intdata.M[3].Dim(1)){ // trg mat-vec
            size_t interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[3];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+2];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                interac_idx++;
              }
            }

            Matrix<Real_t>::GEMM(vbuff[4],vbuff[3],intdata.M[2]);
            Matrix<Real_t>::GEMM(vbuff[5],vbuff[4],intdata.M[3]);

            interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[5];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+3];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                { // Add vbuff[5] to trg_value
                  size_t vdim=vbuff[5].Dim(1);
                  assert(trg_value.Dim(1)==vdim);
                  for(size_t i=0;i<vdim;i++) trg_value[0][i]+=vbuff[5][interac_idx][i];
                }
                interac_idx++;
              }
            }
          }

          trg0+=trg1_max;
        }
      }
    }
    if(device) MIC_Lock::release_lock(lock_idx);
  }
  #ifdef __INTEL_OFFLOAD
  if(SYNC){
    #pragma offload if(device) target(mic:0)
    {if(device) MIC_Lock::wait_lock(lock_idx);}
  }
  #endif
  Profile::Toc();
}


template <class FMMNode>
void FMM_Pts<FMMNode>::X_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_s2l;
    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[1];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[1];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim()) &&  nodes_in [i]->IsLeaf ()) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) &&  nodes_out[i]->pt_cnt[1]                                          && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->src_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->src_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->surf_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->surf_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=tree->dnwd_check_surf[((FMMNode*)nodes[i])->Depth()];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode*)nodes[i])->FMMData())->dnward_equiv;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);

    size_t m=this->MultipoleOrder();
    size_t Nsrf=(6*(m-1)*(m-1)+2);
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        if(tnode->IsLeaf() && tnode->pt_cnt[1]<=Nsrf){ // skip: handled in U-list
          interac_cnt.push_back(0);
          continue;
        }
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->Depth());

        size_t interac_cnt_=0;
        { // X_Type
          Mat_Type type=X_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(0+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(0+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(0+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::X_List     (SetupData<Real_t>&  setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add X_List contribution.
  this->EvalListPts(setup_data, device);
}


template <class FMMNode>
void FMM_Pts<FMMNode>::W_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_m2t;
    setup_data. input_data=&buff[0];
    setup_data.output_data=&buff[5];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[0];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && nodes_in [i]->pt_cnt[0]                                                            ) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=tree->upwd_equiv_surf[((FMMNode*)nodes[i])->Depth()];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode*)nodes[i])->FMMData())->upward_equiv;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      coord.dsp[i]=0;
      coord.cnt[i]=0;
      value.dsp[i]=0;
      value.cnt[i]=0;
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);

    size_t m=this->MultipoleOrder();
    size_t Nsrf=(6*(m-1)*(m-1)+2);
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->Depth());

        size_t interac_cnt_=0;
        { // W_Type
          Mat_Type type=W_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0){ // Is non-leaf ghost node
            }else if(snode->IsLeaf() && snode->pt_cnt[0]<=Nsrf) continue; // skip: handled in U-list
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.25*s-(0+0.25*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.25*s-(0+0.25*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.25*s-(0+0.25*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::W_List     (SetupData<Real_t>&  setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add W_List contribution.
  this->EvalListPts(setup_data, device);
}


template <class FMMNode>
void FMM_Pts<FMMNode>::U_ListSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_s2t;
    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[5];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim()) && nodes_in [i]->IsLeaf()                            ) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) && (nodes_out[i]->trg_coord.Dim()                                  ) && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->src_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->src_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->surf_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->surf_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);

    size_t m=this->MultipoleOrder();
    size_t Nsrf=(6*(m-1)*(m-1)+2);
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->Depth());

        size_t interac_cnt_=0;
        { // U0_Type
          Mat_Type type=U0_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // U1_Type
          Mat_Type type=U1_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*1.0*s-(scoord[0]+0.5*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*1.0*s-(scoord[1]+0.5*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*1.0*s-(scoord[2]+0.5*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // U2_Type
          Mat_Type type=U2_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // X_Type
          Mat_Type type=X_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          if(tnode->pt_cnt[1]<=Nsrf)
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // W_Type
          Mat_Type type=W_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0) continue; // Is non-leaf ghost node
            if(snode->pt_cnt[0]> Nsrf) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::U_List     (SetupData<Real_t>&  setup_data, bool device){
  //Add U_List contribution.
  this->EvalListPts(setup_data, device);
}


template <class FMMNode>
void FMM_Pts<FMMNode>::Down2TargetSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_l2t;
    setup_data. input_data=&buff[1];
    setup_data.output_data=&buff[5];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[1];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->Depth()==level || level==-1) && nodes_in [i]->trg_coord.Dim() && nodes_in [i]->IsLeaf() && !nodes_in [i]->IsGhost()) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->Depth()==level || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=tree->dnwd_equiv_surf[((FMMNode*)nodes[i])->Depth()];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode*)nodes[i])->FMMData())->dnward_equiv;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      coord.dsp[i]=0;
      coord.cnt[i]=0;
      value.dsp[i]=0;
      value.cnt[i]=0;
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);
    if(this->ScaleInvar()){ // Set scal
      const Kernel<Real_t>* ker=kernel->k_l2l;
      for(size_t l=0;l<MAX_DEPTH;l++){ // scal[l*4+0]
        Vector<Real_t>& scal=data.interac_data.scal[l*4+0];
        Vector<Real_t>& scal_exp=ker->trg_scal;
        scal.ReInit(scal_exp.Dim());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
        }
      }
      for(size_t l=0;l<MAX_DEPTH;l++){ // scal[l*4+1]
        Vector<Real_t>& scal=data.interac_data.scal[l*4+1];
        Vector<Real_t>& scal_exp=ker->src_scal;
        scal.ReInit(scal_exp.Dim());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
        }
      }
    }

    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid];

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->Depth());

        size_t interac_cnt_=0;
        { // D2T_Type
          Mat_Type type=D2T_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->Depth());
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(0+0.5*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(0+0.5*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(0+0.5*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
    { // Set M[0], M[1]
      InteracData& interac_data=data.interac_data;
      pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
      pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
      if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
        data.interac_data.M[0]=this->mat->Mat(level, DC2DE0_Type, 0);
        data.interac_data.M[1]=this->mat->Mat(level, DC2DE1_Type, 0);
      }else{
        data.interac_data.M[0].ReInit(0,0);
        data.interac_data.M[1].ReInit(0,0);
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode>
void FMM_Pts<FMMNode>::Down2Target(SetupData<Real_t>&  setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add Down2Target contribution.
  this->EvalListPts(setup_data, device);
}


template <class FMMNode>
void FMM_Pts<FMMNode>::PostProcessing(FMMTree_t* tree, std::vector<FMMNode_t*>& nodes, BoundaryType bndry){
}


template <class FMMNode>
void FMM_Pts<FMMNode>::CopyOutput(FMMNode** nodes, size_t n){
}

}//end namespace
