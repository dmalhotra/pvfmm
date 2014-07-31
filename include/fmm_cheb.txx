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
#include <mem_utils.hpp>
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
          for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=V_Type;
          for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=V1_Type;
          for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=U2U_Type;
          for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=D2D_Type;
          for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
          for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
            Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
            M.Resize(0,0);
          }
          type=D2T_Type;
          for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
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
void FMM_Cheb<FMMNode>::Initialize(int mult_order, int cheb_deg_, const MPI_Comm& comm_, const Kernel<Real_t>* kernel_, const Kernel<Real_t>* aux_kernel_){
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
  FMM_Pts<FMMNode>::Initialize(mult_order, comm_, kernel_, aux_kernel_);
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
        for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=V_Type;
        for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=V1_Type;
        for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=U2U_Type;
        for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=D2D_Type;
        for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
        for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
          Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
          M.Resize(0,0);
        }
        type=D2T_Type;
        for(int l=-BC_LEVELS;l<MAX_DEPTH;l++)
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


template <class FMMNode>
Permutation<typename FMMNode::Real_t>& FMM_Cheb<FMMNode>::PrecompPerm(Mat_Type type, Perm_Type perm_indx){
  int dim=3; //Only supporting 3D

  //Check if the matrix already exists.
  Permutation<Real_t>& P_ = FMM_Pts<FMMNode>::PrecompPerm(type, perm_indx);
  if(P_.Dim()!=0) return P_;

  Matrix<size_t> swap_xy(10,9);
  Matrix<size_t> swap_xz(10,9);
  { // This is repeated from FMM_Pts::PrecompPerm, but I dont see any other way.
      for(int i=0;i<9;i++)
      for(int j=0;j<9;j++){
        swap_xy[i][j]=j;
        swap_xz[i][j]=j;
      }
      swap_xy[3][0]=1; swap_xy[3][1]=0; swap_xy[3][2]=2;
      swap_xz[3][0]=2; swap_xz[3][1]=1; swap_xz[3][2]=0;


      swap_xy[6][0]=1; swap_xy[6][1]=0; swap_xy[6][2]=2;
      swap_xy[6][3]=4; swap_xy[6][4]=3; swap_xy[6][5]=5;

      swap_xz[6][0]=2; swap_xz[6][1]=1; swap_xz[6][2]=0;
      swap_xz[6][3]=5; swap_xz[6][4]=4; swap_xz[6][5]=3;


      swap_xy[9][0]=4; swap_xy[9][1]=3; swap_xy[9][2]=5;
      swap_xy[9][3]=1; swap_xy[9][4]=0; swap_xy[9][5]=2;
      swap_xy[9][6]=7; swap_xy[9][7]=6; swap_xy[9][8]=8;

      swap_xz[9][0]=8; swap_xz[9][1]=7; swap_xz[9][2]=6;
      swap_xz[9][3]=5; swap_xz[9][4]=4; swap_xz[9][5]=3;
      swap_xz[9][6]=2; swap_xz[9][7]=1; swap_xz[9][8]=0;
  }

  //Compute the matrix.
  Permutation<Real_t> P;
  switch (type){

    case UC2UE_Type:
    {
      break;
    }
    case DC2DE_Type:
    {
      break;
    }
    case S2U_Type:
    {
      break;
    }
    case U2U_Type:
    {
      break;
    }
    case D2D_Type:
    {
      break;
    }
    case D2T_Type:
    {
      break;
    }
    case U0_Type:
    {
      int coeff_cnt=((cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6;
      int n3=(int)pow((Real_t)(cheb_deg+1),dim);
      int dof=(perm_indx<C_Perm?this->kernel.ker_dim[0]:this->kernel.ker_dim[1]);
      size_t p_indx=perm_indx % C_Perm;
      Permutation<Real_t> P0(n3*dof);
      if(dof%3==0 && this->kernel.ker_name.compare("biot_savart")==0) //biot_savart
        for(int j=0;j<dof;j++)
        for(int i=0;i<n3;i++)
          P0.scal[i+n3*j]*=(perm_indx<C_Perm?1:-1);
      if(p_indx==ReflecX || p_indx==ReflecY || p_indx==ReflecZ){
        for(int j=0;j<dof;j++)
        for(int i=0;i<n3;i++){
          int x[3]={i%(cheb_deg+1), (i/(cheb_deg+1))%(cheb_deg+1), i/(cheb_deg+1)/(cheb_deg+1)};
          P0.scal[i+n3*j]*=(x[p_indx]%2?-1.0:1.0);
          if(dof%3==0) //stokes_vel (and like kernels)
            P0.scal[i+n3*j]*=( j   %3==p_indx?-1.0:1.0);
          if(dof%3==0 && (dof/3)%3==0)
            P0.scal[i+n3*j]*=((j/3)%3==p_indx?-1.0:1.0);
        }
      }else if(p_indx==SwapXY || p_indx==SwapXZ){
        int indx[3];
        if(p_indx==SwapXY) {indx[0]=1; indx[1]=0; indx[2]=2;}
        if(p_indx==SwapXZ) {indx[0]=2; indx[1]=1; indx[2]=0;}
        for(int j=0;j<dof;j++)
        for(int i=0;i<n3;i++){
          int x[3]={i%(cheb_deg+1), (i/(cheb_deg+1))%(cheb_deg+1), i/(cheb_deg+1)/(cheb_deg+1)};
          P0.perm[i+n3*j]=x[indx[0]]+(x[indx[1]]+x[indx[2]]*(cheb_deg+1))*(cheb_deg+1)
                          +n3*(p_indx==SwapXY?swap_xy[dof][j]:swap_xz[dof][j]);
        }
      }

      std::vector<size_t> coeff_map(n3*dof,0);
      {
        int indx=0;
        for(int j=0;j<dof;j++)
        for(int i=0;i<n3;i++){
          int x[3]={i%(cheb_deg+1), (i/(cheb_deg+1))%(cheb_deg+1), i/(cheb_deg+1)/(cheb_deg+1)};
          if(x[0]+x[1]+x[2]<=cheb_deg){
            coeff_map[i+n3*j]=indx;
            indx++;
          }
        }
      }
      P=Permutation<Real_t>(coeff_cnt*dof);
      {
        int indx=0;
        for(int j=0;j<dof;j++)
        for(int i=0;i<n3;i++){
          int x[3]={i%(cheb_deg+1), (i/(cheb_deg+1))%(cheb_deg+1), i/(cheb_deg+1)/(cheb_deg+1)};
          if(x[0]+x[1]+x[2]<=cheb_deg){
            P.perm[indx]=coeff_map[P0.perm[i+n3*j]];
            P.scal[indx]=          P0.scal[i+n3*j] ;
            indx++;
          }
        }
      }
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
    case V_Type:
    {
      break;
    }
    case V1_Type:
    {
      break;
    }
    case W_Type:
    {
      if(perm_indx>=C_Perm) P=PrecompPerm(U0_Type, perm_indx);
      else P=PrecompPerm(D2D_Type, perm_indx);
      break;
    }
    case X_Type:
    {
      if(perm_indx<C_Perm) P=PrecompPerm(U0_Type, perm_indx);
      else P=PrecompPerm(D2D_Type, perm_indx);
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
  if(this->Homogen()) level=0;

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
      Real_t r=pow(0.5,level);
      Real_t c[3]={0,0,0};

      // Coord of upward check surface
      std::vector<Real_t> uc_coord=u_check_surf(this->MultipoleOrder(),c,level);
      size_t n_uc=uc_coord.size()/3;

      // Evaluate potential at check surface.
      Matrix<Real_t> M_s2c(n_src*this->aux_kernel.ker_dim[0],n_uc*this->aux_kernel.ker_dim[1]); //source 2 check
      Matrix<Real_t> M_s2c_local(n_src*this->aux_kernel.ker_dim[0],n_uc*this->aux_kernel.ker_dim[1]);
      {
        M_s2c.SetZero();
        M_s2c_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_uc;i+=np){
          std::vector<Real_t> M_=cheb_integ(cheb_deg, &uc_coord[i*3], r, this->aux_kernel);
          #ifdef __VERBOSE__
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_uc+100*cnt_done*np)/(class_count*n_uc)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->aux_kernel.ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2c.Dim(0); j++)
              M_s2c_local[j][i*this->aux_kernel.ker_dim[1]+k] = M_[j+k*M_s2c.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2c_local[0], M_s2c[0], M_s2c.Dim(0)*M_s2c.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      Matrix<Real_t>& M_c2e = this->Precomp(level, UC2UE_Type, 0);
      M=M_s2c*M_c2e;
      break;
    }
    case D2T_Type:
    {
      if(this->MultipoleOrder()==0) break;
      Matrix<Real_t>& M_s2t=FMM_Pts<FMMNode>::Precomp(level, type, mat_indx);
      int n_trg=M_s2t.Dim(1)/this->kernel.ker_dim[1];

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel.ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel.ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel.ker_dim[1],M[j]);
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
      Real_t s=pow(0.5,level);
      int* coord=this->interac_list.RelativeCoord(type,mat_indx);
      Real_t coord_diff[3]={(coord[0]-1)*s*0.5,(coord[1]-1)*s*0.5,(coord[2]-1)*s*0.5};
      std::vector<Real_t>& rel_trg_coord = this->mat->RelativeTrgCoord();
      size_t n_trg = rel_trg_coord.size()/3;
      std::vector<Real_t> trg_coord(n_trg*3);
      for(size_t j=0;j<n_trg;j++){
        trg_coord[j*3+0]=rel_trg_coord[j*3+0]*s-coord_diff[0];
        trg_coord[j*3+1]=rel_trg_coord[j*3+1]*s-coord_diff[1];
        trg_coord[j*3+2]=rel_trg_coord[j*3+2]*s-coord_diff[2];
      }

      // Evaluate potential at target points.
      Matrix<Real_t> M_s2t(n_src*this->kernel.ker_dim [0], n_trg*this->kernel.ker_dim [1]);
      Matrix<Real_t> M_s2t_local(n_src*this->kernel.ker_dim [0], n_trg*this->kernel.ker_dim [1]);
      {
        M_s2t.SetZero();
        M_s2t_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> s2t=cheb_integ(cheb_deg, &trg_coord[i*3], (Real_t)(s*2.0), this->kernel);
          #ifdef __VERBOSE__
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel.ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++)
              M_s2t_local[j][i*this->kernel.ker_dim[1]+k] = s2t[j+k*M_s2t.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2t_local[0], M_s2t[0], M_s2t.Dim(0)*M_s2t.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel.ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel.ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel.ker_dim[1],M[j]);
      }
      break;
    }
    case U1_Type:
    {
      // Coord of target points
      Real_t s=pow(0.5,level);
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
      Matrix<Real_t> M_s2t(n_src*this->kernel.ker_dim [0], n_trg*this->kernel.ker_dim [1]);
      Matrix<Real_t> M_s2t_local(n_src*this->kernel.ker_dim [0], n_trg*this->kernel.ker_dim [1]);
      {
        M_s2t.SetZero();
        M_s2t_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> s2t=cheb_integ(cheb_deg, &trg_coord[i*3], s, this->kernel);
          #ifdef __VERBOSE__
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel.ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++)
              M_s2t_local[j][i*this->kernel.ker_dim[1]+k] = s2t[j+k*M_s2t.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2t_local[0], M_s2t[0], M_s2t.Dim(0)*M_s2t.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel.ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel.ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel.ker_dim[1],M[j]);
      }
      break;
    }
    case U2_Type:
    {
      // Coord of target points
      Real_t s=pow(0.5,level);
      int* coord=this->interac_list.RelativeCoord(type,mat_indx);
      Real_t coord_diff[3]={(coord[0]+1)*s*0.25,(coord[1]+1)*s*0.25,(coord[2]+1)*s*0.25};
      std::vector<Real_t>& rel_trg_coord = this->mat->RelativeTrgCoord();
      size_t n_trg = rel_trg_coord.size()/3;
      std::vector<Real_t> trg_coord(n_trg*3);
      for(size_t j=0;j<n_trg;j++){
        trg_coord[j*3+0]=rel_trg_coord[j*3+0]*s-coord_diff[0];
        trg_coord[j*3+1]=rel_trg_coord[j*3+1]*s-coord_diff[1];
        trg_coord[j*3+2]=rel_trg_coord[j*3+2]*s-coord_diff[2];
      }

      // Evaluate potential at target points.
      Matrix<Real_t> M_s2t(n_src*this->kernel.ker_dim [0], n_trg*this->kernel.ker_dim [1]);
      Matrix<Real_t> M_s2t_local(n_src*this->kernel.ker_dim [0], n_trg*this->kernel.ker_dim [1]);
      {
        M_s2t.SetZero();
        M_s2t_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> s2t=cheb_integ(cheb_deg, &trg_coord[i*3], (Real_t)(s*0.5), this->kernel);
          #ifdef __VERBOSE__
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->kernel.ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++)
              M_s2t_local[j][i*this->kernel.ker_dim[1]+k] = s2t[j+k*M_s2t.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_s2t_local[0], M_s2t[0], M_s2t.Dim(0)*M_s2t.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel.ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel.ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel.ker_dim[1],M[j]);
      }
      break;
    }
    case W_Type:
    {
      if(this->MultipoleOrder()==0) break;
      Matrix<Real_t>& M_s2t=FMM_Pts<FMMNode>::Precomp(level, type, mat_indx);
      int n_trg=M_s2t.Dim(1)/this->kernel.ker_dim[1];

      // Compute Chebyshev approx from target potential.
      M.Resize(M_s2t.Dim(0), n_src*this->kernel.ker_dim [1]);
      #pragma omp parallel for schedule(dynamic)
      for(size_t j=0; j<(size_t)M_s2t.Dim(0); j++){
        Matrix<Real_t> M_trg(n_trg,this->kernel.ker_dim[1],M_s2t[j],false);
        M_trg=M_trg.Transpose();
        cheb_approx<Real_t,Real_t>(M_s2t[j],cheb_deg,this->kernel.ker_dim[1],M[j]);
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
      Real_t s=pow(0.5,level-1);
      int* coord=this->interac_list.RelativeCoord(type,mat_indx);
      Real_t c[3]={-(coord[0]-1)*s*0.25,-(coord[1]-1)*s*0.25,-(coord[2]-1)*s*0.25};
      std::vector<Real_t> trg_coord=d_check_surf(this->MultipoleOrder(),c,level);
      size_t n_trg=trg_coord.size()/3;

      // Evaluate potential at target points.
      Matrix<Real_t> M_xs2c(n_src*this->aux_kernel.ker_dim[0], n_trg*this->aux_kernel.ker_dim[1]);
      Matrix<Real_t> M_xs2c_local(n_src*this->aux_kernel.ker_dim[0], n_trg*this->aux_kernel.ker_dim[1]);
      {
        M_xs2c.SetZero();
        M_xs2c_local.SetZero();
        size_t cnt_done=0;
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=myrank;i<n_trg;i+=np){
          std::vector<Real_t> M_=cheb_integ(cheb_deg, &trg_coord[i*3], s, this->aux_kernel);
          #ifdef __VERBOSE__
          #pragma omp critical
          if(!myrank){
            cnt_done++;
            std::cout<<"\r Progress: "<<(100*progress*n_trg+100*cnt_done*np)/(class_count*n_trg)<<"% "<<std::flush;
          }
          #endif
          for(int k=0; k<this->aux_kernel.ker_dim[1]; k++)
            for(size_t j=0; j<(size_t)M_xs2c.Dim(0); j++)
              M_xs2c_local[j][i*this->aux_kernel.ker_dim[1]+k] = M_[j+k*M_xs2c.Dim(0)];
        }
        if(!myrank) std::cout<<"\r                    \r"<<std::flush;
        MPI_Allreduce(M_xs2c_local[0], M_xs2c[0], M_xs2c.Dim(0)*M_xs2c.Dim(1), par::Mpi_datatype<Real_t>::value(), par::Mpi_datatype<Real_t>::sum(), this->comm);
      }
      Matrix<Real_t>& M_c2e = this->Precomp(level, DC2DE_Type, 0);
      M=M_xs2c*M_c2e;
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
void FMM_Cheb<FMMNode>::CollectNodeData(std::vector<FMMNode*>& node, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, std::vector<size_t> extra_size){
  if(      buff.size()<6)       buff.resize(6);
  if(    n_list.size()<6)     n_list.resize(6);
  if(extra_size.size()<6) extra_size.resize(6,0);

  size_t n_coeff=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
  if(node.size()==0) return;
  {// 4. cheb_in
    int indx=4;
    int dof=this->kernel.ker_dim[0];
    size_t vec_sz=dof*n_coeff;
    std::vector< FMMNode* > node_lst;
    for(size_t i=0;i<node.size();i++)
      if(node[i]->IsLeaf())
        node_lst.push_back(node[i]);
    n_list[indx]=node_lst;
    extra_size[indx]+=node_lst.size()*vec_sz;

    #pragma omp parallel for
    for(size_t i=0;i<node.size();i++){ // Move data before resizing buff[indx]
      Vector<Real_t>& cheb_in =node[i]->ChebData();
      Vector<Real_t> new_buff=cheb_in;
      cheb_in.Swap(new_buff);
    }
  }
  {// 5. cheb_out
    int indx=5;
    int dof=this->kernel.ker_dim[1];
    size_t vec_sz=dof*n_coeff;
    std::vector< FMMNode* > node_lst;
    for(size_t i=0;i<node.size();i++)
      if(node[i]->IsLeaf() && !node[i]->IsGhost())
        node_lst.push_back(node[i]);
    n_list[indx]=node_lst;
    extra_size[indx]+=node_lst.size()*vec_sz;

    #pragma omp parallel for
    for(size_t i=0;i<node.size();i++){ // Move data before resizing buff[indx]
      Vector<Real_t>& cheb_out=((FMMData*)node[i]->FMMData())->cheb_out;
      cheb_out.ReInit(0);
    }
  }
  FMM_Pts<FMMNode>::CollectNodeData(node, buff, n_list, extra_size);
  {// 4. cheb_in
    int indx=4;
    int dof=this->kernel.ker_dim[0];
    size_t vec_sz=dof*n_coeff;
    Vector< FMMNode* >& node_lst=n_list[indx];
    Real_t* buff_ptr=buff[indx][0]+buff[indx].Dim(0)*buff[indx].Dim(1)-extra_size[indx];
    #pragma omp parallel for
    for(size_t i=0;i<node_lst.Dim();i++){
      Vector<Real_t>& cheb_in =node_lst[i]->ChebData();
      mem::memcopy(buff_ptr+i*vec_sz, &cheb_in [0], cheb_in .Dim()*sizeof(Real_t));
      cheb_in .ReInit(vec_sz, buff_ptr+i*vec_sz, false);
      //if(node_lst[i]->IsGhost()) cheb_in .SetZero();
    }
    buff[indx].AllocDevice(true);
  }
  {// 5. cheb_out
    int indx=5;
    int dof=this->kernel.ker_dim[1];
    size_t vec_sz=dof*n_coeff;
    Vector< FMMNode* >& node_lst=n_list[indx];
    Real_t* buff_ptr=buff[indx][0]+buff[indx].Dim(0)*buff[indx].Dim(1)-extra_size[indx];
    #pragma omp parallel for
    for(size_t i=0;i<node_lst.Dim();i++){
      Vector<Real_t>& cheb_out=((FMMData*)node_lst[i]->FMMData())->cheb_out;
      cheb_out.ReInit(vec_sz, buff_ptr+i*vec_sz, false);
      cheb_out.SetZero();
    }
    buff[indx].AllocDevice(true);
  }
}


template <class FMMNode>
void FMM_Cheb<FMMNode>::Source2UpSetup(SetupData<Real_t>& setup_data, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(this->MultipoleOrder()==0) return;
  FMM_Pts<FMMNode>::Source2UpSetup(setup_data, buff, n_list, level, device);

  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=&this->aux_kernel;
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
  //Add Source2Up contribution.
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::X_ListSetup(SetupData<Real_t>& setup_data, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(this->MultipoleOrder()==0) return;
  FMM_Pts<FMMNode>::X_ListSetup(setup_data, buff, n_list, level, device);

  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=&this->aux_kernel;
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
    if(this->dev_buffer.Dim()<n) this->dev_buffer.Resize(n);
  }
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::X_List     (SetupData<Real_t>& setup_data, bool device){
  //Add X_List contribution.
  FMM_Pts<FMMNode>::X_List(setup_data, device);
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::W_ListSetup(SetupData<Real_t>& setup_data, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(this->MultipoleOrder()==0) return;
  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=&this->kernel;
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
    if(this->dev_buffer.Dim()<n) this->dev_buffer.Resize(n);
  }
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::W_List     (SetupData<Real_t>& setup_data, bool device){
  //Add W_List contribution.
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::U_ListSetup(SetupData<Real_t>& setup_data, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  FMM_Pts<FMMNode>::U_ListSetup(setup_data, buff, n_list, level, device);

  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=&this->kernel;
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
    if(this->dev_buffer.Dim()<n) this->dev_buffer.Resize(n);
  }
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::U_List     (SetupData<Real_t>& setup_data, bool device){
  //Add U_List contribution.
  FMM_Pts<FMMNode>::U_List(setup_data, device);
  this->EvalList(setup_data, device);
}



template <class FMMNode>
void FMM_Cheb<FMMNode>::Down2TargetSetup(SetupData<Real_t>& setup_data, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device){
  if(this->MultipoleOrder()==0) return;
  { // Set setup_data
    setup_data.level=level;
    setup_data.kernel=&this->kernel;
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
  //Add Down2Target contribution.
  this->EvalList(setup_data, device);
}

template <class FMMNode>
void FMM_Cheb<FMMNode>::PostProcessing(std::vector<FMMNode_t*>& nodes){
  size_t n=nodes.size();
  #pragma omp parallel
  {
    int omp_p=omp_get_num_threads();
    int pid = omp_get_thread_num();
    size_t a=(pid*n)/omp_p;
    size_t b=((pid+1)*n)/omp_p;

    //For each node, compute interaction from point sources.
    for(size_t i=a;i<b;i++){
      Vector<Real_t>& trg_coord=nodes[i]->trg_coord;
      Vector<Real_t>& trg_value=nodes[i]->trg_value;
      Vector<Real_t>& cheb_out =((FMMData*)nodes[i]->FMMData())->cheb_out;

      //Initialize target potential.
      size_t trg_cnt=trg_coord.Dim()/COORD_DIM;
      //trg_value.assign(trg_cnt*dof*this->kernel.ker_dim[1],0);

      //Sample local expansion at target points.
      if(trg_cnt>0 && cheb_out.Dim()>0){
        Real_t* c=nodes[i]->Coord();
        Real_t scale=pow(2.0,nodes[i]->Depth()+1);
        std::vector<Real_t> rel_coord(COORD_DIM*trg_cnt);
        for(size_t j=0;j<trg_cnt;j++) for(int k=0;k<COORD_DIM;k++)
          rel_coord[j*COORD_DIM+k]=(trg_coord[j*COORD_DIM+k]-c[k])*scale-1.0;
        cheb_eval(cheb_out, cheb_deg, rel_coord, trg_value);
      }
    }
  }
}

}//end namespace
