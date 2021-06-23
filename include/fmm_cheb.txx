/**
 * \file fmm_cheb.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 3-07-2011
 * \brief This file contains the implementation of the FMM_Cheb class.
 */

#include <omp.h>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#ifdef PVFMM_HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#include <dtypes.h>
#include <parUtils.h>
#include <cheb_utils.hpp>
#include <mem_mgr.hpp>
#include <profile.hpp>

namespace pvfmm{

template <class FMMNode>
FMM_Cheb<FMMNode>::~FMM_Cheb() {
  if(this->mat!=NULL){
    int rank;
    MPI_Comm_rank(this->comm,&rank);
    if(!rank){
      FILE* f=fopen(this->mat_fname.c_str(),"r");
      if(f==NULL) { //File does not exists.
        { // Delete easy to compute matrices.
          Mat_Type type=W_Type;
          for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=V_Type;
          for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=V1_Type;
          for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=U2U_Type;
          for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=D2D_Type;
          for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=D2T_Type;
          for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
        }
        this->mat->Save2File(this->mat_fname.c_str());
      }else fclose(f);
    }
  }
}


template <class FMMNode>
void FMM_Cheb<FMMNode>::Initialize(int mult_order, int cheb_deg_, const MPI_Comm& comm_, const Kernel<Real_t>* kernel_){
  Profile::Tic("InitFMM_Cheb",&comm_,true);{
  int rank;
  MPI_Comm_rank(comm_,&rank);

  int dim=3; //Only supporting 3D
  cheb_deg=cheb_deg_;
  if(this->mat_fname.size()==0){
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
    st<<"Precomp_"<<kernel_->ker_name.c_str()<<"_q"<<cheb_deg<<"_m"<<mult_order;
    if(sizeof(Real_t)==8) st<<"";
    else if(sizeof(Real_t)==4) st<<"_f";
    else st<<"_t"<<sizeof(Real_t);
    st<<".data";
    this->mat_fname=st.str();
  }
  if(!rank){
    FILE* f=fopen(this->mat_fname.c_str(),"r");
    if(f==NULL) { //File does not exists.
      std::cout<<"Could not find precomputed data file for "<<kernel_->ker_name.c_str()<<" kernel with q="<<cheb_deg<<" and m="<<mult_order<<".\n";
      std::cout<<"This data will be computed and stored for future use at:\n"<<this->mat_fname<<'\n';
      std::cout<<"This may take a while...\n";
    }else fclose(f);
  }
  //this->mat->LoadFile(this->mat_fname.c_str(), this->comm);
  FMM_Pts<FMMNode>::Initialize(mult_order, comm_, kernel_);
  this->mat->RelativeTrgCoord()=cheb_nodes<Real_t>(ChebDeg(),dim);

  Profile::Tic("PrecompD2T",&this->comm,false,4);
  this->PrecompAll(D2T_Type);
  Profile::Toc();

  //Volume solver.
  Profile::Tic("PrecompS2M",&this->comm,false,4);
  this->PrecompAll(S2U_Type);
  Profile::Toc();

  Profile::Tic("PrecompX",&this->comm,false,4);
  this->PrecompAll(X_Type);
  Profile::Toc();

  Profile::Tic("PrecompW",&this->comm,false,4);
  this->PrecompAll(W_Type);
  Profile::Toc();

  Profile::Tic("PrecompU0",&this->comm,false,4);
  this->PrecompAll(U0_Type);
  Profile::Toc();

  Profile::Tic("PrecompU1",&this->comm,false,4);
  this->PrecompAll(U1_Type);
  Profile::Toc();

  Profile::Tic("PrecompU2",&this->comm,false,4);
  this->PrecompAll(U2_Type);
  Profile::Toc();

  Profile::Tic("Save2File",&this->comm,false,4);
  if(!rank){
    FILE* f=fopen(this->mat_fname.c_str(),"r");
    if(f==NULL) { //File does not exists.
      { // Delete easy to compute matrices.
        Mat_Type type=W_Type;
        for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=V_Type;
        for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=V1_Type;
        for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=U2U_Type;
        for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=D2D_Type;
        for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=D2T_Type;
        for(int l=-PVFMM_BC_LEVELS;l<PVFMM_MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
      }
      this->mat->Save2File(this->mat_fname.c_str());
    }else fclose(f);
  }
  Profile::Toc();

  Profile::Tic("Recompute",&this->comm,false,4);
  { // Recompute matrices.
    this->PrecompAll(W_Type);
    this->PrecompAll(V_Type);
    this->PrecompAll(V1_Type);
    this->PrecompAll(U2U_Type);
    this->PrecompAll(D2D_Type);
    this->PrecompAll(D2T_Type);
  }
  Profile::Toc();

  }Profile::Toc();
}


template <class Real_t>
Permutation<Real_t> cheb_perm(size_t q, size_t p_indx, const Permutation<Real_t>& ker_perm, const Vector<Real_t>* scal_exp=NULL){
  int dim=3; //Only supporting 3D
  int dof=ker_perm.Dim();

  int coeff_cnt=((q+1)*(q+2)*(q+3))/6;
  int n3=pvfmm::pow<unsigned int>(q+1,dim);

  Permutation<Real_t> P0(n3*dof);
  if(scal_exp && p_indx==Scaling){ // Set level-by-level scaling
    assert(dof==scal_exp->Dim());
    Vector<Real_t> scal(scal_exp->Dim());
    for(size_t i=0;i<scal.Dim();i++){
      scal[i]=pvfmm::pow<Real_t>(2.0,(*scal_exp)[i]);
    }
    for(int j=0;j<dof;j++){
      for(int i=0;i<n3;i++){
        P0.scal[j*n3+i]*=scal[j];
      }
    }
  }
  { // Set P0.scal
    for(int j=0;j<dof;j++){
      for(int i=0;i<n3;i++){
        P0.scal[j*n3+i]*=ker_perm.scal[j];
      }
    }
  }
  if(p_indx==ReflecX || p_indx==ReflecY || p_indx==ReflecZ){
    for(int j=0;j<dof;j++)
    for(int i=0;i<n3;i++){
      size_t x[3]={i%(q+1), (i/(q+1))%(q+1), i/(q+1)/(q+1)};
      P0.scal[i+n3*j]*=(x[p_indx-ReflecX]%2?-1.0:1.0);
    }
  }

  { // Set P0.perm
    int indx[3]={0,1,2};
    if(p_indx==SwapXY) {indx[0]=1; indx[1]=0; indx[2]=2;}
    if(p_indx==SwapXZ) {indx[0]=2; indx[1]=1; indx[2]=0;}
    for(int j=0;j<dof;j++)
    for(int i=0;i<n3;i++){
      size_t x[3]={i%(q+1), (i/(q+1))%(q+1), i/(q+1)/(q+1)};
      P0.perm[i+n3*j]=x[indx[0]]+(x[indx[1]]+x[indx[2]]*(q+1))*(q+1)
                      +n3*ker_perm.perm[j];
    }
  }

  std::vector<size_t> coeff_map(n3*dof,0);
  {
    int indx=0;
    for(int j=0;j<dof;j++)
    for(int i=0;i<n3;i++){
      size_t x[3]={i%(q+1), (i/(q+1))%(q+1), i/(q+1)/(q+1)};
      if(x[0]+x[1]+x[2]<=q){
        coeff_map[i+n3*j]=indx;
        indx++;
      }
    }
  }
  Permutation<Real_t> P=Permutation<Real_t>(coeff_cnt*dof);
  {
    int indx=0;
    for(int j=0;j<dof;j++)
    for(int i=0;i<n3;i++){
      size_t x[3]={i%(q+1), (i/(q+1))%(q+1), i/(q+1)/(q+1)};
      if(x[0]+x[1]+x[2]<=q){
        P.perm[indx]=coeff_map[P0.perm[i+n3*j]];
        P.scal[indx]=          P0.scal[i+n3*j] ;
        indx++;
      }
    }
  }

  return P;
}

template <class FMMNode>
Permutation<typename FMMNode::Real_t>& FMM_Cheb<FMMNode>::PrecompPerm(Mat_Type type, Perm_Type perm_indx){
  //int dim=3; //Only supporting 3D
  Real_t eps=1e-10;

  //Check if the matrix already exists.
  Permutation<Real_t>& P_ = FMM_Pts<FMMNode>::PrecompPerm(type, perm_indx);
  if(P_.Dim()!=0) return P_;

  size_t q=cheb_deg;
  size_t m=this->MultipoleOrder();
  size_t p_indx=perm_indx % C_Perm;

  //Compute the matrix.
  Permutation<Real_t> P;
  switch (type){

    case S2U_Type:
    {
      Vector<Real_t> scal_exp;
      Permutation<Real_t> ker_perm;
      if(perm_indx<C_Perm){
        ker_perm=this->kernel->k_s2m->perm_vec[0     +p_indx];
        scal_exp=this->kernel->k_s2m->src_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]-=PVFMM_COORD_DIM;
        P=cheb_perm(q, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }else{
        ker_perm=this->kernel->k_m2m->perm_vec[0     +p_indx];
        scal_exp=this->kernel->k_m2m->src_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];

        // Check that target perm for the two kernels agree.
        Permutation<Real_t> ker_perm0=this->kernel->k_s2m->perm_vec[C_Perm+p_indx];
        Permutation<Real_t> ker_perm1=this->kernel->k_m2m->perm_vec[C_Perm+p_indx];
        assert(ker_perm0.Dim()==ker_perm1.Dim());
        if(ker_perm0.Dim()>0 && pvfmm::fabs<Real_t>(ker_perm0.scal[0]-ker_perm1.scal[0])>eps){
          for(size_t i=0;i<ker_perm0.Dim();i++){
            ker_perm0.scal[i]*=-1.0;
          }
          for(size_t i=0;i<ker_perm.Dim();i++){
            ker_perm.scal[i]*=-1.0;
          }
        }
        for(size_t i=0;i<ker_perm0.Dim();i++){
          assert(                   (ker_perm0.perm[i]-ker_perm1.perm[i])== 0);
          assert(pvfmm::fabs<Real_t>(ker_perm0.scal[i]-ker_perm1.scal[i])<eps);
        }

        Real_t s=0;
        // Check that target scaling for the two kernels agree.
        const Vector<Real_t>& scal_exp0=this->kernel->k_s2m->trg_scal;
        const Vector<Real_t>& scal_exp1=this->kernel->k_m2m->trg_scal;
        assert(scal_exp0.Dim()>0 && scal_exp0.Dim()==scal_exp1.Dim());
        if(scal_exp0.Dim()>0){
          s=(scal_exp0[0]-scal_exp1[0]);
          for(size_t i=1;i<scal_exp0.Dim();i++){
            assert(pvfmm::fabs<Real_t>(s-(scal_exp0[i]-scal_exp1[i]))<eps);
            // In general this is not necessary, but to allow this, we must
            // also change src_scal accordingly.
          }
        }

        // Apply the difference in scaling of the two kernels.
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]+=s;
        P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }
      break;
    }
    case D2T_Type:
    {
      Vector<Real_t> scal_exp;
      Permutation<Real_t> ker_perm;
      if(perm_indx<C_Perm){ // Source permutation
        ker_perm=this->kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=this->kernel->k_l2l->trg_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];

        // Check that source perm for the two kernels agree.
        Permutation<Real_t> ker_perm0=this->kernel->k_l2t->perm_vec[0     +p_indx];
        Permutation<Real_t> ker_perm1=this->kernel->k_l2l->perm_vec[0     +p_indx];
        assert(ker_perm0.Dim()==ker_perm1.Dim());
        if(ker_perm0.Dim()>0 && pvfmm::fabs<Real_t>(ker_perm0.scal[0]-ker_perm1.scal[0])>eps){
          for(size_t i=0;i<ker_perm0.Dim();i++){
            ker_perm0.scal[i]*=-1.0;
          }
          for(size_t i=0;i<ker_perm.Dim();i++){
            ker_perm.scal[i]*=-1.0;
          }
        }
        for(size_t i=0;i<ker_perm0.Dim();i++){
          assert(                   (ker_perm0.perm[i]-ker_perm1.perm[i])== 0);
          assert(pvfmm::fabs<Real_t>(ker_perm0.scal[i]-ker_perm1.scal[i])<eps);
        }

        Real_t s=0;
        // Check that source scaling for the two kernels agree.
        const Vector<Real_t>& scal_exp0=this->kernel->k_l2t->src_scal;
        const Vector<Real_t>& scal_exp1=this->kernel->k_l2l->src_scal;
        assert(scal_exp0.Dim()>0 && scal_exp0.Dim()==scal_exp1.Dim());
        if(scal_exp0.Dim()>0){
          s=(scal_exp0[0]-scal_exp1[0]);
          for(size_t i=1;i<scal_exp0.Dim();i++){
            assert(pvfmm::fabs<Real_t>(s-(scal_exp0[i]-scal_exp1[i]))<eps);
            // In general this is not necessary, but to allow this, we must
            // also change trg_scal accordingly.
          }
        }

        // Apply the difference in scaling of the two kernels.
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]+=s;
        P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }else{ // Target permutation
        ker_perm=this->kernel->k_l2t->perm_vec[C_Perm+p_indx];
        scal_exp=this->kernel->k_l2t->trg_scal;
        P=cheb_perm(q, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }
      break;
    }
    case U0_Type:
    {
      Vector<Real_t> scal_exp;
      Permutation<Real_t> ker_perm;
      if(perm_indx<C_Perm){
        ker_perm=this->kernel->k_s2t->perm_vec[0     +p_indx];
        scal_exp=this->kernel->k_s2t->src_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]-=PVFMM_COORD_DIM;
      }else{
        ker_perm=this->kernel->k_s2t->perm_vec[C_Perm+p_indx];
        scal_exp=this->kernel->k_s2t->trg_scal;
      }
      P=cheb_perm(q, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      break;
    }
    case U1_Type:
    {
      P=PrecompPerm(U0_Type, perm_indx);
      break;
    }
    case U2_Type:
    {
      P=PrecompPerm(U0_Type, perm_indx);
      break;
    }
    case W_Type:
    {
      Vector<Real_t> scal_exp;
      Permutation<Real_t> ker_perm;
      if(perm_indx<C_Perm){ // Source permutation
        ker_perm=this->kernel->k_m2t->perm_vec[0     +p_indx];
        scal_exp=this->kernel->k_m2t->src_scal;
        P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }else{ // Target permutation
        ker_perm=this->kernel->k_m2t->perm_vec[C_Perm+p_indx];
        scal_exp=this->kernel->k_m2t->trg_scal;
        P=cheb_perm(q, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }
      break;
    }
    case X_Type:
    {
      Vector<Real_t> scal_exp;
      Permutation<Real_t> ker_perm;
      if(perm_indx<C_Perm){
        ker_perm=this->kernel->k_s2l->perm_vec[0     +p_indx];
        scal_exp=this->kernel->k_s2l->src_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]-=PVFMM_COORD_DIM;
        P=cheb_perm(q, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }else{
        ker_perm=this->kernel->k_s2l->perm_vec[C_Perm+p_indx];
        scal_exp=this->kernel->k_s2l->trg_scal;
        P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
      }
      break;
    }
    default:
      return FMM_Pts<FMMNode>::PrecompPerm(type, perm_indx);
      break;
  }

  //Save the matrix for future use.
  #pragma omp critical (PRECOMP_MATRIX_PTS)
  if(P_.Dim()==0){ P_=P;}

  return P_;
}


template <class FMMNode>
Matrix<typename FMMNode::Real_t>& FMM_Cheb<FMMNode>::Precomp(int level, Mat_Type type, size_t mat_indx){
  if(this->ScaleInvar()) level=0;

  //Check if the matrix already exists.
  //Compute matrix from symmetry class (if possible).
  Matrix<Real_t>& M_ = this->mat->Mat(level, type, mat_indx);
  if(M_.Dim(0)!=0 && M_.Dim(1)!=0) return M_;
  else{ //Compute matrix from symmetry class (if possible).
    size_t class_indx = this->interac_list.InteracClass(type, mat_indx);
    if(class_indx!=mat_indx){
      Matrix<Real_t>& M0 = this->Precomp(level, type, class_indx);
      Permutation<Real_t>& Pr = this->interac_list.Perm_R(level, type, mat_indx);
      Permutation<Real_t>& Pc = this->interac_list.Perm_C(level, type, mat_indx);
      if(Pr.Dim()>0 && Pc.Dim()>0 && M0.Dim(0)>0 && M0.Dim(1)>0) return M_;
    }
  }

  int myrank, np;
  MPI_Comm_rank(this->comm, &myrank);
  MPI_Comm_size(this->comm,&np);

  size_t progress=0, class_count=0;
  { // Determine precomputation progress.
    size_t mat_cnt=this->interac_list.ListCount((Mat_Type)type);
    for(size_t i=0; i<mat_cnt; i++){
      size_t indx=this->interac_list.InteracClass((Mat_Type)type,i);
      if(indx==i){
        class_count++;
        if(i<mat_indx) progress++;
      }
    }
  }

  //Compute the matrix.
  Matrix<Real_t> M;
  int n_src=((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6;
  switch (type){

    case S2U_Type:
    {
      if(this->MultipoleOrder()==0) break;
      Real_t r=pvfmm::pow<Real_t>(0.5,level);
      Real_t c[3]={0,0,0};

      // Coord of upward check surface
      std::vector<Real_t> uc_coord=u_check_surf(this->MultipoleOrder(),c,level);
      size_t n_uc=uc_coord.size()/3;

      // Evaluate potential at check surface.
      Matrix<Real_t> M_s2c(n_src*this->kernel->k_s2m->ker_dim[0],n_uc*this->kernel->k_s2m->ker_dim[1]); //source 2 check
      Matrix<Real_t> M_s2c_local(n_src*this->kernel->k_s2m->ker_dim[0],n_uc*this->kernel->k_s2m->ker_dim[1]);
      {
        M_s2c.SetZero();
        M_s2c_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_uc;i+=np){
          std::vector<Real_t> M_=cheb_integ(cheb_deg, &uc_coord[i*3], r, *this->kernel->k_s2m);
          #ifdef PVFMM_VERBOSE
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_uc+100*cnt_done*np)/(class_count*n_uc)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel->k_s2m->ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2c.Dim(0); j++)
              M_s2c_local[j][i*this->kernel->k_s2m->ker_dim[1]+k] = M_[j+k*M_s2c.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2c_local[0], M_s2c[0], M_s2c.Dim(0)*M_s2c.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      Matrix<Real_t>& M_c2e0 = this->Precomp(level, UC2UE0_Type, 0);
      Matrix<Real_t>& M_c2e1 = this->Precomp(level, UC2UE1_Type, 0);
      M=(M_s2c*M_c2e0)*M_c2e1;
      break;
    }
    case D2T_Type:
    {
      if(this->MultipoleOrder()==0) break;
      Matrix<Real_t>& M_s2t=FMM_Pts<FMMNode>::Precomp(level, type, mat_indx);
      int n_trg=M_s2t.Dim(1)/this->kernel->k_l2t->ker_dim[1];

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel->k_l2t->ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel->k_l2t->ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel->k_l2t->ker_dim[1],M[j]);
      }
      #pragma omp critical (PRECOMP_MATRIX_PTS)
      {
        M_s2t.Resize(0,0);
      }
      break;
    }
    case U0_Type:
    {
      // Coord of target points
      Real_t s=pvfmm::pow<Real_t>(0.5,level);
      int* coord=this->interac_list.RelativeCoord(type,mat_indx);
      Real_t coord_diff[3]={(Real_t)((coord[0]-1)*s*0.5),(Real_t)((coord[1]-1.0)*s*0.5),(Real_t)((coord[2]-1.0)*s*0.5)};
      std::vector<Real_t>& rel_trg_coord = this->mat->RelativeTrgCoord();
      size_t n_trg = rel_trg_coord.size()/3;
      std::vector<Real_t> trg_coord(n_trg*3);
      for(size_t j=0;j<n_trg;j++){
        trg_coord[j*3+0]=rel_trg_coord[j*3+0]*s-coord_diff[0];
        trg_coord[j*3+1]=rel_trg_coord[j*3+1]*s-coord_diff[1];
        trg_coord[j*3+2]=rel_trg_coord[j*3+2]*s-coord_diff[2];
      }

      // Evaluate potential at target points.
      Matrix<Real_t> M_s2t(n_src*this->kernel->k_s2t->ker_dim [0], n_trg*this->kernel->k_s2t->ker_dim [1]);
      Matrix<Real_t> M_s2t_local(n_src*this->kernel->k_s2t->ker_dim [0], n_trg*this->kernel->k_s2t->ker_dim [1]);
      {
        M_s2t.SetZero();
        M_s2t_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> s2t=cheb_integ(cheb_deg, &trg_coord[i*3], (Real_t)(s*2.0), *this->kernel->k_s2t);
          #ifdef PVFMM_VERBOSE
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel->k_s2t->ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++)
              M_s2t_local[j][i*this->kernel->k_s2t->ker_dim[1]+k] = s2t[j+k*M_s2t.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2t_local[0], M_s2t[0], M_s2t.Dim(0)*M_s2t.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel->k_s2t->ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel->k_s2t->ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel->k_s2t->ker_dim[1],M[j]);
      }
      break;
    }
    case U1_Type:
    {
      // Coord of target points
      Real_t s=pvfmm::pow<Real_t>(0.5,level);
      int* coord=this->interac_list.RelativeCoord(type,mat_indx);
      Real_t coord_diff[3]={coord[0]*s,coord[1]*s,coord[2]*s};
      std::vector<Real_t>& rel_trg_coord = this->mat->RelativeTrgCoord();
      size_t n_trg = rel_trg_coord.size()/3;
      std::vector<Real_t> trg_coord(n_trg*3);
      for(size_t j=0;j<n_trg;j++){
        trg_coord[j*3+0]=rel_trg_coord[j*3+0]*s-coord_diff[0];
        trg_coord[j*3+1]=rel_trg_coord[j*3+1]*s-coord_diff[1];
        trg_coord[j*3+2]=rel_trg_coord[j*3+2]*s-coord_diff[2];
      }

      // Evaluate potential at target points.
      Matrix<Real_t> M_s2t(n_src*this->kernel->k_s2t->ker_dim [0], n_trg*this->kernel->k_s2t->ker_dim [1]);
      Matrix<Real_t> M_s2t_local(n_src*this->kernel->k_s2t->ker_dim [0], n_trg*this->kernel->k_s2t->ker_dim [1]);
      {
        M_s2t.SetZero();
        M_s2t_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> s2t=cheb_integ(cheb_deg, &trg_coord[i*3], s, *this->kernel->k_s2t);
          #ifdef PVFMM_VERBOSE
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel->k_s2t->ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++)
              M_s2t_local[j][i*this->kernel->k_s2t->ker_dim[1]+k] = s2t[j+k*M_s2t.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2t_local[0], M_s2t[0], M_s2t.Dim(0)*M_s2t.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel->k_s2t->ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel->k_s2t->ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel->k_s2t->ker_dim[1],M[j]);
      }
      break;
    }
    case U2_Type:
    {
      // Coord of target points
      Real_t s=pvfmm::pow<Real_t>(0.5,level);
      int* coord=this->interac_list.RelativeCoord(type,mat_indx);
      Real_t coord_diff[3]={(Real_t)((coord[0]+1)*s*0.25),(Real_t)((coord[1]+1)*s*0.25),(Real_t)((coord[2]+1)*s*0.25)};
      std::vector<Real_t>& rel_trg_coord = this->mat->RelativeTrgCoord();
      size_t n_trg = rel_trg_coord.size()/3;
      std::vector<Real_t> trg_coord(n_trg*3);
      for(size_t j=0;j<n_trg;j++){
        trg_coord[j*3+0]=rel_trg_coord[j*3+0]*s-coord_diff[0];
        trg_coord[j*3+1]=rel_trg_coord[j*3+1]*s-coord_diff[1];
        trg_coord[j*3+2]=rel_trg_coord[j*3+2]*s-coord_diff[2];
      }

      // Evaluate potential at target points.
      Matrix<Real_t> M_s2t(n_src*this->kernel->k_s2t->ker_dim [0], n_trg*this->kernel->k_s2t->ker_dim [1]);
      Matrix<Real_t> M_s2t_local(n_src*this->kernel->k_s2t->ker_dim [0], n_trg*this->kernel->k_s2t->ker_dim [1]);
      {
        M_s2t.SetZero();
        M_s2t_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> s2t=cheb_integ(cheb_deg, &trg_coord[i*3], (Real_t)(s*0.5), *this->kernel->k_s2t);
          #ifdef PVFMM_VERBOSE
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel->k_s2t->ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++)
              M_s2t_local[j][i*this->kernel->k_s2t->ker_dim[1]+k] = s2t[j+k*M_s2t.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2t_local[0], M_s2t[0], M_s2t.Dim(0)*M_s2t.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel->k_s2t->ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel->k_s2t->ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel->k_s2t->ker_dim[1],M[j]);
      }
      break;
    }
    case W_Type:
    {
      if(this->MultipoleOrder()==0) break;
      Matrix<Real_t>& M_s2t=FMM_Pts<FMMNode>::Precomp(level, type, mat_indx);
      int n_trg=M_s2t.Dim(1)/this->kernel->k_m2t->ker_dim[1];

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel->k_m2t->ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel->k_m2t->ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel->k_m2t->ker_dim[1],M[j]);
      }
      #pragma omp critical (PRECOMP_MATRIX_PTS)
      {
        M_s2t.Resize(0,0);
      }
      break;
    }
    case X_Type:
    {
      if(this->MultipoleOrder()==0) break;
      // Coord of target points
      Real_t s=pvfmm::pow<Real_t>(0.5,level-1);
      int* coord=this->interac_list.RelativeCoord(type,mat_indx);
      Real_t c[3]={-(Real_t)((coord[0]-1)*s*0.25),-(Real_t)((coord[1]-1)*s*0.25),-(Real_t)((coord[2]-1)*s*0.25)};
      std::vector<Real_t> trg_coord=d_check_surf(this->MultipoleOrder(),c,level);
      size_t n_trg=trg_coord.size()/3;

      // Evaluate potential at target points.
      Matrix<Real_t> M_xs2c(n_src*this->kernel->k_s2l->ker_dim[0], n_trg*this->kernel->k_s2l->ker_dim[1]);
      Matrix<Real_t> M_xs2c_local(n_src*this->kernel->k_s2l->ker_dim[0], n_trg*this->kernel->k_s2l->ker_dim[1]);
      {
        M_xs2c.SetZero();
        M_xs2c_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> M_=cheb_integ(cheb_deg, &trg_coord[i*3], s, *this->kernel->k_s2l);
          #ifdef PVFMM_VERBOSE
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel->k_s2l->ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_xs2c.Dim(0); j++)
              M_xs2c_local[j][i*this->kernel->k_s2l->ker_dim[1]+k] = M_[j+k*M_xs2c.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_xs2c_local[0], M_xs2c[0], M_xs2c.Dim(0)*M_xs2c.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }
      M=M_xs2c;
      break;
    }
    default:
    {
      return FMM_Pts<FMMNode>::Precomp(level, type, mat_indx);
      break;
    }
  }

  //Save the matrix for future use.
  #pragma omp critical (PRECOMP_MATRIX_PTS)
  if(M_.Dim(0)==0 && M_.Dim(1)==0){ M_=M;}

  return M_;
}


template <class FMMNode>
void FMM_Cheb<FMMNode>::CollectNodeData(FMMTree_t* tree, std::vector<FMMNode*>& node, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, std::vector<std::vector<Vector<Real_t>* > > vec_list){
  if(vec_list.size()<6) vec_list.resize(6);
  size_t n_coeff=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
  if(node.size()==0) return;
  {// 4. cheb_in
    int indx=4;
    size_t vec_sz=this->kernel->ker_dim[0]*n_coeff;
    for(size_t i=0;i<node.size();i++){
      if(node[i]->IsLeaf()){
        Vector<Real_t>& data_vec=node[i]->ChebData();
        if(data_vec.Dim()!=vec_sz) data_vec.ReInit(vec_sz);
        vec_list[indx].push_back(&data_vec);
      }else{
        node[i]->ChebData().ReInit(0);
      }
    }
  }
  {// 5. cheb_out
    int indx=5;
    size_t vec_sz=this->kernel->ker_dim[1]*n_coeff;
    for(size_t i=0;i<node.size();i++){
      if(node[i]->IsLeaf() && !node[i]->IsGhost()){
        Vector<Real_t>& data_vec=((FMMData*)node[i]->FMMData())->cheb_out;
        if(data_vec.Dim()!=vec_sz) data_vec.ReInit(vec_sz);
        vec_list[indx].push_back(&data_vec);
      }else{
        ((FMMData*)node[i]->FMMData())->cheb_out.ReInit(0);
      }
    }
  }
  { // Set pt_cnt
    size_t m=this->MultipoleOrder();
    size_t Nsrf=(6*(m-1)*(m-1)+2);
    #pragma omp parallel for
    for(size_t i=0;i<node.size();i++){
      node[i]->pt_cnt[0]+=2*Nsrf;
      node[i]->pt_cnt[1]+=2*Nsrf;
    }
  }
  FMM_Pts<FMMNode_t>::CollectNodeData(tree, node, buff, n_list, vec_list);
}


template <class FMMNode>
void FMM_Cheb<FMMNode>::Source2UpSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  FMM_Pts<FMMNode>::Source2UpSetup(setup_data, tree, buff, n_list, level, device);

  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=this->kernel->k_s2m;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=S2U_Type;

    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[0];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[0];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if(nodes_in [i]->Depth()==level   || level==-1) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if(nodes_out[i]->Depth()==level   || level==-1) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(          ((FMMNode*)nodes_in [i])           )->ChebData()  );
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)nodes_out[i])->FMMData())->upward_equiv);

  this->SetupInterac(setup_data,device);
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::Source2Up     (SetupData<Real_t>& setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add Source2Up contribution.
  FMM_Pts<FMMNode>::Source2Up(setup_data, device);
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::X_ListSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  FMM_Pts<FMMNode>::X_ListSetup(setup_data, tree, buff, n_list, level, device);

  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=this->kernel->k_s2l;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=X_Type;

    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[1];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[1];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if(nodes_in [i]->Depth()==level-1 || level==-1) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if(nodes_out[i]->Depth()==level   || level==-1) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(          ((FMMNode*)nodes_in [i])           )->ChebData()  );
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)nodes_out[i])->FMMData())->dnward_equiv);

  this->SetupInterac(setup_data,device);
  { // Resize device buffer
    size_t n=setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1)*sizeof(Real_t);
    if(this->dev_buffer.Dim()<n) this->dev_buffer.ReInit(n);
  }
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::X_List     (SetupData<Real_t>& setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add X_List contribution.
  FMM_Pts<FMMNode>::X_List(setup_data, device);
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::W_ListSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=this->kernel->k_m2t;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=W_Type;

    setup_data. input_data=&buff[0];
    setup_data.output_data=&buff[5];
    Vector<FMMNode_t*>& nodes_in =n_list[0];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if(nodes_in [i]->Depth()==level+1 || level==-1) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if(nodes_out[i]->Depth()==level   || level==-1) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode*)nodes_in [i])->FMMData())->upward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)nodes_out[i])->FMMData())->cheb_out    );

  this->SetupInterac(setup_data,device);
  { // Resize device buffer
    size_t n=setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1)*sizeof(Real_t);
    if(this->dev_buffer.Dim()<n) this->dev_buffer.ReInit(n);
  }
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::W_List     (SetupData<Real_t>& setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add W_List contribution.
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::U_ListSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  FMM_Pts<FMMNode>::U_ListSetup(setup_data, tree, buff, n_list, level, device);

  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=this->kernel->k_s2t;
    setup_data.interac_type.resize(3);
    setup_data.interac_type[0]=U0_Type;
    setup_data.interac_type[1]=U1_Type;
    setup_data.interac_type[2]=U2_Type;

    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[5];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((level-1<=nodes_in [i]->Depth() && nodes_in [i]->Depth()<=level+1) || level==-1) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((                                  nodes_out[i]->Depth()==level  ) || level==-1) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(          ((FMMNode*)nodes_in [i])           )->ChebData());
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)nodes_out[i])->FMMData())->cheb_out  );

  this->SetupInterac(setup_data,device);
  { // Resize device buffer
    size_t n=setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1)*sizeof(Real_t);
    if(this->dev_buffer.Dim()<n) this->dev_buffer.ReInit(n);
  }
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::U_List     (SetupData<Real_t>& setup_data, bool device){
  //Add U_List contribution.
  FMM_Pts<FMMNode>::U_List(setup_data, device);
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::Down2TargetSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=this->kernel->k_l2t;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=D2T_Type;

    setup_data. input_data=&buff[1];
    setup_data.output_data=&buff[5];
    Vector<FMMNode_t*>& nodes_in =n_list[1];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if(nodes_in [i]->Depth()==level   || level==-1) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if(nodes_out[i]->Depth()==level   || level==-1) setup_data.nodes_out.push_back(nodes_out[i]);
  }

  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode*)nodes_in [i])->FMMData())->dnward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode*)nodes_out[i])->FMMData())->cheb_out    );

  this->SetupInterac(setup_data,device);
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::Down2Target     (SetupData<Real_t>& setup_data, bool device){
  if(!this->MultipoleOrder()) return;
  //Add Down2Target contribution.
  this->EvalList(setup_data, device);
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::PostProcessing(FMMTree_t* tree, std::vector<FMMNode_t*>& nodes, BoundaryType bndry){
#ifndef PVFMM_EXTENDED_BC
  if(this->kernel->k_m2l->vol_poten && bndry==Periodic && PVFMM_BC_LEVELS>0){ // Add analytical near-field to target potential
    const Kernel<Real_t>& k_m2t=*this->kernel->k_m2t;
    int ker_dim[2]={k_m2t.ker_dim[0],k_m2t.ker_dim[1]};

    Vector<Real_t>& up_equiv=((FMMData*)tree->RootNode()->FMMData())->upward_equiv;
    Matrix<Real_t> avg_density(1,ker_dim[0]); avg_density.SetZero();
    for(size_t i0=0;i0<up_equiv.Dim();i0+=ker_dim[0]){
      for(size_t i1=0;i1<ker_dim[0];i1++){
        avg_density[0][i1]+=up_equiv[i0+i1];
      }
    }

    std::vector<Real_t> node_pts0=cheb_nodes<Real_t>(cheb_deg, PVFMM_COORD_DIM);
    int Ncoeff=((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6;
    int Npts=node_pts0.size()/PVFMM_COORD_DIM;
    size_t n=nodes.size();
    #pragma omp parallel
    { // Add volume potential
      int omp_p=omp_get_num_threads();
      int pid=omp_get_thread_num();
      size_t a=((pid+0)*n)/omp_p;
      size_t b=((pid+1)*n)/omp_p;

      Vector<Real_t> node_pts(Npts * PVFMM_COORD_DIM);
      Vector<Real_t> M_vol(Npts * ker_dim[0] * ker_dim[1]);
      Vector<Real_t> vol_poten_coeff(Ncoeff * ker_dim[1]);
      Vector<Real_t> vol_poten(Npts * ker_dim[1]);
      for(size_t i=a;i<b;i++){
        Vector<Real_t>& cheb_out =((FMMData*)nodes[i]->FMMData())->cheb_out;
        if(cheb_out.Dim()>0){
          Real_t* c = nodes[i]->Coord();
          Real_t s = pvfmm::pow<Real_t>(0.5,nodes[i]->Depth());
          for(size_t j=0;j<Npts;j++){
            for(size_t k=0;k<PVFMM_COORD_DIM;k++){
              node_pts[j*PVFMM_COORD_DIM+k] = c[k] + node_pts0[j*PVFMM_COORD_DIM+k] * s;
            }
          }

          vol_poten.SetZero();
          k_m2t.vol_poten(&node_pts[0],Npts,&M_vol[0]);
          for(int j=0;j<Npts;j++){
            for(int k0=0;k0<ker_dim[0];k0++){
              for(int k1=0;k1<ker_dim[1];k1++){
                vol_poten[k1 * Npts + j] += M_vol[(k0 * Npts + j) * ker_dim[1] + k1] * avg_density[0][k0];
              }
            }
          }

          assert(cheb_out.Dim() == vol_poten_coeff.Dim());
          cheb_approx<Real_t, Real_t>(&vol_poten[0], cheb_deg, ker_dim[1], &vol_poten_coeff[0]);
          for(int j=0;j<vol_poten_coeff.Dim();j++) cheb_out[j]-=vol_poten_coeff[j];
        }
      }
    }
  }
#endif

  size_t n=nodes.size();
  #pragma omp parallel
  {
    int omp_p=omp_get_num_threads();
    int pid = omp_get_thread_num();
    size_t a=(pid*n)/omp_p;
    size_t b=((pid+1)*n)/omp_p;

    std::vector<Real_t> tmp_vec; // pre-allocate
    for(size_t i=a;i<b;i++){
      Vector<Real_t>& trg_coord=nodes[i]->trg_coord;
      Vector<Real_t>& trg_value=nodes[i]->trg_value;
      Vector<Real_t>& cheb_out =((FMMData*)nodes[i]->FMMData())->cheb_out;

      //Evaluate potential at target points.
      size_t trg_cnt=trg_coord.Dim()/PVFMM_COORD_DIM;
      if(trg_cnt>0 && cheb_out.Dim()>0){
        Real_t* c=nodes[i]->Coord();
        Real_t scale=pvfmm::pow<Real_t>(2.0,nodes[i]->Depth()+1);
        std::vector<Real_t>& rel_coord=tmp_vec;
        rel_coord.resize(PVFMM_COORD_DIM*trg_cnt);
        for(size_t j=0;j<trg_cnt;j++) for(int k=0;k<PVFMM_COORD_DIM;k++)
          rel_coord[j*PVFMM_COORD_DIM+k]=(trg_coord[j*PVFMM_COORD_DIM+k]-c[k])*scale-1.0;
        cheb_eval(cheb_out, cheb_deg, rel_coord, trg_value);
      }
    }
  }

  FMM_Pts<FMMNode>::PostProcessing(tree,nodes,bndry);
}

}//end namespace
