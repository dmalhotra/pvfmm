#include <fmm_gll.hpp>

#include <iostream>
#include <mpi.h>
#include <omp.h>

#include <pvfmm_common.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <cheb_utils.hpp>
#include <vector.hpp>
#include <cheb_node.hpp>

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;
typedef FMM_Mat_t::FMMData FMM_Data_t;

extern "C" {

  void fmm_gll_init(FMM_GLL_t* fmm_data, int gll_order_, int cheb_order_, int multipole_order, MPI_Comm comm){
    int np, myrank;
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm, &myrank);

    fmm_data->comm=comm;
    fmm_data->gll_order=gll_order_-1;
    fmm_data->cheb_order=cheb_order_-1;
    fmm_data->multipole_order=multipole_order;

    int gll_order=fmm_data->gll_order;
    int cheb_order=fmm_data->cheb_order;
    { // Get GLL node points.
      fmm_data->gll_nodes=new std::vector<double>(gll_order+1);
      std::vector<double>& gll_nodes=*((std::vector<double>*)fmm_data->gll_nodes);
      std::vector<double> w(gll_order+1);
      pvfmm::gll_quadrature(gll_order, &gll_nodes[0], &w[0]);
    }

    for(int ker_indx=0;ker_indx<2;ker_indx++)
    { // Initialize Matrices
      FMM_Mat_t* fmm_mat;
      pvfmm::Kernel<double>* mykernel;
      FMM_Tree_t* mytree;

      if(ker_indx==0){
        pvfmm::Profile::Tic("Init:Biot-Savart",&fmm_data->comm,true,3);
        fmm_data->fmm_mat_biotsavart=new FMM_Mat_t;
        fmm_mat=((FMM_Mat_t*)fmm_data->fmm_mat_biotsavart);

        fmm_data->kernel_biotsavart=&pvfmm::ker_biot_savart;
        mykernel=(pvfmm::Kernel<double>*)fmm_data->kernel_biotsavart;

        fmm_data->tree_biotsavart=new FMM_Tree_t(fmm_data->comm);
        mytree=((FMM_Tree_t*)fmm_data->tree_biotsavart);
      }else{
        pvfmm::Profile::Tic("Init:LaplaceGrad",&fmm_data->comm,true,3);
        fmm_data->fmm_mat_laplace_grad=new FMM_Mat_t;
        fmm_mat=((FMM_Mat_t*)fmm_data->fmm_mat_laplace_grad);

        fmm_data->kernel_laplace_grad=&pvfmm::LaplaceKernel<double>::grad_ker();
        mykernel=(pvfmm::Kernel<double>*)fmm_data->kernel_laplace_grad;

        fmm_data->tree_laplace_grad=new FMM_Tree_t(fmm_data->comm);
        mytree=((FMM_Tree_t*)fmm_data->tree_laplace_grad);
      }

      //Various parameters.
      FMMNode_t::NodeData mydata;
      mydata.dim=COORD_DIM;
      mydata.data_dof=mykernel->ker_dim[0];
      mydata.max_pts=1; // Points per octant.
      mydata.cheb_deg=cheb_order;
      mydata.tol=1e-10;

      //Set source coordinates and values.
      //mydata.pt_coord=point_distrib(UnifGrid,np*8,fmm_data->comm);
      {
        size_t N=np*8;
        size_t NN=(size_t)pow((double)N,1.0/3.0);
        size_t N_total=NN*NN*NN;
        size_t start= myrank   *N_total/np;
        size_t end  =(myrank+1)*N_total/np;
        mydata.pt_coord.Resize((end-start)*3);
        for(size_t i=start;i<end;i++){
          mydata.pt_coord[(i-start)*3+0]=(((double)((i/  1    )%NN)+0.5)/NN);
          mydata.pt_coord[(i-start)*3+1]=(((double)((i/ NN    )%NN)+0.5)/NN);
          mydata.pt_coord[(i-start)*3+2]=(((double)((i/(NN*NN))%NN)+0.5)/NN);
        }
      }
      mydata.pt_value.Resize((mydata.pt_coord.Dim()/3)*mydata.data_dof);
      mydata.cheb_coord=mydata.pt_coord;
      mydata.cheb_value=mydata.pt_value;
      mydata.input_fn=NULL;

      //Initialize FMM_Mat.
      pvfmm::Profile::Tic("InitMat",&fmm_data->comm,true,5);
      fmm_mat->Initialize(fmm_data->multipole_order,mydata.cheb_deg,fmm_data->comm,mykernel);
      pvfmm::Profile::Toc();

      //Create Tree and initialize with input data.
      pvfmm::Profile::Tic("Tree Construction",&fmm_data->comm,true,5);
      mytree->Initialize(&mydata);
      pvfmm::Profile::Toc();

      //Initialize FMM Tree
      pvfmm::Profile::Tic("Init FMM Tree",&fmm_data->comm,true,5);
      pvfmm::BoundaryType bndry=pvfmm::Periodic;//pvfmm::FreeSpace;
      mytree->InitFMM_Tree(false,bndry);
      pvfmm::Profile::Toc();

#ifndef NDEBUG
      //Check Tree.
      pvfmm::Profile::Tic("Check Tree",&fmm_data->comm,true,5);
      mytree->CheckTree();
      pvfmm::Profile::Toc();

      pvfmm::Profile::Tic("FMM Eval",&fmm_data->comm,true,1);
      mytree->SetupFMM(fmm_mat);
      mytree->RunFMM();
      pvfmm::Profile::Toc();
#endif

      pvfmm::Profile::Toc();
    }

#ifndef NDEBUG
    pvfmm::Profile::print(&fmm_data->comm);
#endif
  }

  void fmm_gll_free(FMM_GLL_t* fmm_data){
    fmm_data->gll_order=-1;
    fmm_data->cheb_order=-1;
    fmm_data->multipole_order=-1;

    fmm_data->kernel_biotsavart=NULL;
    delete (FMM_Mat_t*)fmm_data->fmm_mat_biotsavart; fmm_data->fmm_mat_biotsavart=NULL;
    delete (FMM_Tree_t*)fmm_data->tree_biotsavart; fmm_data->tree_biotsavart=NULL;

    fmm_data->kernel_laplace_grad=NULL;
    delete (FMM_Mat_t*)fmm_data->fmm_mat_laplace_grad; fmm_data->fmm_mat_laplace_grad=NULL;
    delete (FMM_Tree_t*)fmm_data->tree_laplace_grad; fmm_data->tree_laplace_grad=NULL;

    delete (std::vector<double>*)fmm_data->gll_nodes; fmm_data->gll_nodes=NULL;
  }

  void fmm_gll_run(FMM_GLL_t* fmm_data, size_t node_cnt, double* node_coord, unsigned char* node_depth, double** node_gll_data){
    int gll_order=fmm_data->gll_order;
    int cheb_order=fmm_data->cheb_order;
    FMM_Mat_t& fmm_mat=*((FMM_Mat_t*)fmm_data->fmm_mat_biotsavart);
    FMM_Tree_t& mytree=*((FMM_Tree_t*)fmm_data->tree_biotsavart);
    pvfmm::Kernel<double>* mykernel=((pvfmm::Kernel<double>*)fmm_data->kernel_biotsavart);
    pvfmm::BoundaryType bndry=pvfmm::Periodic;//pvfmm::FreeSpace;

    // Find out number of OMP thereads.
    int omp_p=omp_get_max_threads();

    pvfmm::Profile::Tic("Construct Tree",&fmm_data->comm,true,1);
    //Construct pvfmm::MortonId from node_coord and node_depth.
    std::vector<pvfmm::MortonId> nodes(node_cnt);
    double s=0.25/(1UL<<MAX_DEPTH);
    for(size_t i=0;i<node_cnt;i++)
      nodes[i]=pvfmm::MortonId(node_coord[i*3+0]+s,node_coord[i*3+1]+s,node_coord[i*3+2]+s,node_depth[i]);

    // Construct tree from morton id.
    size_t node_iter=0;
    std::vector<FMMNode_t*> tree_nodes;
    FMMNode_t* node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    while(node!=NULL && node_iter<node_cnt){
      pvfmm::MortonId mid=node->GetMortonId();
      if(mid.isAncestor(nodes[node_iter])){
        node->SetGhost(false);
        node->Subdivide();
      }else{
        node->Truncate();
        node->SetGhost(mid!=nodes[node_iter]);
        if(mid==nodes[node_iter]){
          tree_nodes.push_back(node);
          node_iter++;
        }
      }
      node->DataDOF()=mykernel->ker_dim[0];
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }
    assert(node_iter==node_cnt);
    while(node!=NULL){
      node->Truncate();
      node->SetGhost(true);
      node->DataDOF()=mykernel->ker_dim[0];
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }

    // Add GLL data
    size_t n_gll_coeff=(gll_order+1)*(gll_order+2)*(gll_order+3)/6;
    size_t n_cheb_coeff=(cheb_order+1)*(cheb_order+2)*(cheb_order+3)/6;
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      pvfmm::Vector<double> buff;
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        FMMNode_t* node=tree_nodes[j];
        node->DataDOF()=mykernel->ker_dim[0];
        buff.Resize(n_gll_coeff*node->DataDOF());
        pvfmm::gll2cheb<double,double>(&(node_gll_data[j][0]), gll_order, node->DataDOF(), &(buff[0]));

        node->ChebData().Resize(n_cheb_coeff*node->DataDOF());
        double* cheb_out=&(node->ChebData()[0]);
        int indx_in=0;
        int indx_out=0;
        for(int k_=0;k_<node->DataDOF();k_++)
        for(int k0=0;k0      <=gll_order;k0++)
        for(int k1=0;k0+k1   <=gll_order;k1++)
        for(int k2=0;k0+k1+k2<=gll_order;k2++){
          if(k0+k1+k2<=cheb_order){
            cheb_out[indx_out]=buff[indx_in];
            indx_out++;
          }
          indx_in++;
        }

      }
    }

    // Set avg to zero for Periodic boundary condition
    std::vector<pvfmm::Vector<double> > cheb_w(node_cnt); //Copy vorticity.
    if(bndry==pvfmm::Periodic){
      std::vector<double> avg_w(mykernel->ker_dim[0],0);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          avg_w[k]+=node->ChebData()[k*n_cheb_coeff]*pow(0.5,node->Depth()*3);
      }
      std::vector<double> glb_avg_w(mykernel->ker_dim[0],0);
      MPI_Allreduce(&avg_w[0], &glb_avg_w[0], mykernel->ker_dim[0], MPI_DOUBLE, MPI_SUM, fmm_data->comm);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          node->ChebData()[k*n_cheb_coeff]-=glb_avg_w[k];
        cheb_w[j]=node->ChebData();
      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    double max_w=0; // |w|_inf
    int n_gll_pts=(gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[0];
    for(size_t j=0;j<node_cnt;j++)
      for(int k=0; k<n_gll_pts; k++)
        if(max_w<fabs(node_gll_data[j][k]))
          max_w=node_gll_data[j][k];

    //Check Tree.
    pvfmm::Profile::Tic("Check Tree",&fmm_data->comm,true,1);
    mytree.CheckTree();
    pvfmm::Profile::Toc();

    //Report divergence.
    gll_div(fmm_data, node_cnt, node_coord, node_depth, node_gll_data);
#endif

    //FMM Evaluate.
    pvfmm::Profile::Tic("FMM Eval",&fmm_data->comm,true,1);
    mytree.SetupFMM(&fmm_mat);
    mytree.RunFMM();
    pvfmm::Profile::Toc();

    //Copy FMM output to tree Data.
    mytree.Copy_FMMOutput();

    //Set average velocity to zero.
    if(bndry==pvfmm::Periodic){
      std::vector<double> avg_v(mykernel->ker_dim[1],0);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          avg_v[k]+=node->ChebData()[k*n_cheb_coeff]*pow(0.5,node->Depth()*3);
      }
      std::vector<double> glb_avg_v(mykernel->ker_dim[1],0);
      MPI_Allreduce(&avg_v[0], &glb_avg_v[0], mykernel->ker_dim[1], MPI_DOUBLE, MPI_SUM, fmm_data->comm);
      //std::cout<<glb_avg_v[0]<<' '<<glb_avg_v[1]<<' '<<glb_avg_v[2]<<'\n';
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          node->ChebData()[k*n_cheb_coeff]-=glb_avg_v[k];
      }
    }

#ifndef NDEBUG
    // Confirm that 2:1 balance did not change the number of tree nodes.
    size_t new_cnt=0;
    node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    while(node!=NULL){
      if(node->IsLeaf() && !node->IsGhost()) new_cnt++;
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }
    assert(node_cnt==new_cnt);
#endif

    // Construct GLL data from FMM output.
    pvfmm::Profile::Tic("Output",&fmm_data->comm,true,1);
    node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    std::vector<double> corner_val(8*mykernel->ker_dim[1],0);
    std::vector<double> corner_val_loc(8*mykernel->ker_dim[1],0);
    std::vector<double> corner_coord(2); corner_coord[0]=0.0; corner_coord[1]=1.0;
    node->ReadVal(corner_coord,corner_coord,corner_coord, &corner_val_loc[0], false);
    MPI_Allreduce(&corner_val_loc[0], &corner_val[0], 8*mykernel->ker_dim[1], MPI_DOUBLE, MPI_SUM, fmm_data->comm);
    std::vector<double> dx(3*mykernel->ker_dim[1],0);
    for(int i=0;i<mykernel->ker_dim[1];i++){
      dx[0+i*3]=corner_val[1+8*i]-corner_val[0+8*i];
      dx[1+i*3]=corner_val[2+8*i]-corner_val[0+8*i];
      dx[2+i*3]=corner_val[4+8*i]-corner_val[0+8*i];
    }

    std::vector<double>& gll_nodes=*((std::vector<double>*)fmm_data->gll_nodes);
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        pvfmm::Vector<double> gll_data((gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[1],&node_gll_data[j][0],false);
        cheb_eval(tree_nodes[j]->ChebData(), cheb_order, gll_nodes, gll_nodes, gll_nodes, gll_data);

        size_t indx=0;
        double* coord=tree_nodes[j]->Coord();
        double s=pow(0.5,tree_nodes[j]->Depth()+1);
        for(int l=0;l<mykernel->ker_dim[1];l++)
        for(int k2=0;k2<=gll_order;k2++)
        for(int k1=0;k1<=gll_order;k1++)
        for(int k0=0;k0<=gll_order;k0++){
          gll_data[indx]-=(coord[0]+(1.0+gll_nodes[k0])*s-0.5)*dx[0+l*3]+
                          (coord[1]+(1.0+gll_nodes[k1])*s-0.5)*dx[1+l*3]+
                          (coord[2]+(1.0+gll_nodes[k2])*s-0.5)*dx[2+l*3];
          indx++;
        }
      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    //Compute |v|_inf
    double max_v=0;
    for(size_t j=0;j<node_cnt;j++){
      pvfmm::Vector<double> gll_data((gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[1],&node_gll_data[j][0],false);
      for(size_t k=0;k<gll_data.Dim();k++)
        if(max_v<fabs(gll_data[k])) max_v=gll_data[k];
    }

    //Compute |v^2|_l1
    double l2=0;
    pvfmm::Profile::Tic("L2-norm",&fmm_data->comm,true,1);
    std::vector<double> cheb_pts=pvfmm::cheb_nodes<double>(cheb_order*2, 1);
    pvfmm::Vector<double> integ_data((2*cheb_order+1)*(2*cheb_order+1)*(2*cheb_order+1)*mykernel->ker_dim[1]);
    pvfmm::Vector<double> integ_coeff((2*cheb_order+1)*(2*cheb_order+1)*(2*cheb_order+1)*mykernel->ker_dim[1]);
    for(size_t j=0;j<node_cnt;j++){
      double s=pow(0.5,tree_nodes[j]->Depth()*3);
      cheb_eval(tree_nodes[j]->ChebData(), cheb_order, cheb_pts, cheb_pts, cheb_pts, integ_data);

      //Make average zero.
      {
        size_t indx=0;
        double* coord=tree_nodes[j]->Coord();
        double s=pow(0.5,tree_nodes[j]->Depth()+1);
        for(int l=0;l<mykernel->ker_dim[1];l++)
        for(int k2=0;k2<=2*cheb_order;k2++)
        for(int k1=0;k1<=2*cheb_order;k1++)
        for(int k0=0;k0<=2*cheb_order;k0++){
          integ_data[indx]-=(coord[0]+(1.0+cheb_pts[k0])*s-0.5)*dx[0+l*3]+
                            (coord[1]+(1.0+cheb_pts[k1])*s-0.5)*dx[1+l*3]+
                            (coord[2]+(1.0+cheb_pts[k2])*s-0.5)*dx[2+l*3];
          indx++;
        }
      }

      for(size_t i=0;i<integ_data.Dim();i++) integ_data[i]*=integ_data[i];
      pvfmm::cheb_approx<double,double>(&integ_data[0], 2*cheb_order, mykernel->ker_dim[1], &integ_coeff[0]);
      l2+=s*integ_coeff[0*(2*cheb_order+1)*(2*cheb_order+2)*(2*cheb_order+3)/6];
      l2+=s*integ_coeff[1*(2*cheb_order+1)*(2*cheb_order+2)*(2*cheb_order+3)/6];
      l2+=s*integ_coeff[2*(2*cheb_order+1)*(2*cheb_order+2)*(2*cheb_order+3)/6];
    }
    pvfmm::Profile::Toc();

    //Compute |w - curl v|_inf
    double w_err=0;
    pvfmm::Profile::Tic("Curl",&fmm_data->comm,true,1);
    pvfmm::Vector<double> buff0(n_cheb_coeff*mykernel->ker_dim[0]);
    pvfmm::Vector<double> buff1((gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[0]);
    for(size_t j=0;j<node_cnt;j++){
      tree_nodes[j]->Curl();
      for(size_t k=0;k<cheb_w[j].Dim();k++)
        buff0[k]=tree_nodes[j]->ChebData()[k]-cheb_w[j][k];
      cheb_eval(buff0, cheb_order, gll_nodes, gll_nodes, gll_nodes, buff1);
      for(size_t k=0;k<buff1.Dim();k++)
        if(w_err<fabs(buff1[k])) w_err=fabs(buff1[k]);
    }
    pvfmm::Profile::Toc();

    //Display norms.
    double glb_l2=0;
    double glb_max_v=0;
    double glb_max_w=0;
    double glb_max_w_err=0;
    MPI_Allreduce(&l2, &glb_l2, 1, MPI_DOUBLE, MPI_SUM, fmm_data->comm);
    MPI_Allreduce(&max_v, &glb_max_v, 1, MPI_DOUBLE, MPI_MAX, fmm_data->comm);
    MPI_Allreduce(&max_w, &glb_max_w, 1, MPI_DOUBLE, MPI_MAX, fmm_data->comm);
    MPI_Allreduce(&w_err, &glb_max_w_err, 1, MPI_DOUBLE, MPI_MAX, fmm_data->comm);
    int myrank;
    MPI_Comm_rank(fmm_data->comm, &myrank);
    if(!myrank) std::cout<<"[fmm] |v^2|_1          = "<<glb_l2<<'\n';;
    if(!myrank) std::cout<<"[fmm] |v|_inf          = "<<glb_max_v<<'\n';;
    if(!myrank) std::cout<<"[fmm] |w|_inf          = "<<glb_max_w<<'\n';;
    if(!myrank) std::cout<<"[fmm] |w - curl v|_inf = "<<glb_max_w_err<<'\n';
#endif

#ifndef NDEBUG
    pvfmm::Profile::print(&fmm_data->comm);
#endif
  }

  void fmm_gll_laplace_grad(FMM_GLL_t* fmm_data, size_t node_cnt, double* node_coord, unsigned char* node_depth, double** node_gll_data){
    int gll_order=fmm_data->gll_order;
    int cheb_order=fmm_data->cheb_order;
    FMM_Mat_t& fmm_mat=*((FMM_Mat_t*)fmm_data->fmm_mat_laplace_grad);
    FMM_Tree_t& mytree=*((FMM_Tree_t*)fmm_data->tree_laplace_grad);
    pvfmm::Kernel<double>* mykernel=((pvfmm::Kernel<double>*)fmm_data->kernel_laplace_grad);
    pvfmm::BoundaryType bndry=pvfmm::Periodic;//pvfmm::FreeSpace;

    // Find out number of OMP thereads.
    int omp_p=omp_get_max_threads();

    pvfmm::Profile::Tic("Construct Tree",&fmm_data->comm,true,1);
    //Construct pvfmm::MortonId from node_coord and node_depth.
    std::vector<pvfmm::MortonId> nodes(node_cnt);
    double s=0.25/(1UL<<MAX_DEPTH);
    for(size_t i=0;i<node_cnt;i++)
      nodes[i]=pvfmm::MortonId(node_coord[i*3+0]+s,node_coord[i*3+1]+s,node_coord[i*3+2]+s,node_depth[i]);

    // Construct tree from morton id.
    size_t node_iter=0;
    std::vector<FMMNode_t*> tree_nodes;
    FMMNode_t* node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    while(node!=NULL && node_iter<node_cnt){
      pvfmm::MortonId mid=node->GetMortonId();
      if(mid.isAncestor(nodes[node_iter])){
        node->SetGhost(false);
        node->Subdivide();
      }else{
        node->Truncate();
        node->SetGhost(mid!=nodes[node_iter]);
        if(mid==nodes[node_iter]){
          tree_nodes.push_back(node);
          node_iter++;
        }
      }
      node->DataDOF()=mykernel->ker_dim[0];
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }
    assert(node_iter==node_cnt);
    while(node!=NULL){
      node->Truncate();
      node->SetGhost(true);
      node->DataDOF()=mykernel->ker_dim[0];
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }

    // Add GLL data
    size_t n_gll_coeff=(gll_order+1)*(gll_order+2)*(gll_order+3)/6;
    size_t n_cheb_coeff=(cheb_order+1)*(cheb_order+2)*(cheb_order+3)/6;
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      pvfmm::Vector<double> buff;
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        FMMNode_t* node=tree_nodes[j];
        node->DataDOF()=mykernel->ker_dim[0];
        buff.Resize(n_gll_coeff*node->DataDOF());
        pvfmm::gll2cheb<double,double>(&(node_gll_data[j][0]), gll_order, node->DataDOF(), &(buff[0]));

        node->ChebData().Resize(n_cheb_coeff*node->DataDOF());
        double* cheb_out=&(node->ChebData()[0]);
        int indx_in=0;
        int indx_out=0;
        for(int k_=0;k_<node->DataDOF();k_++)
        for(int k0=0;k0      <=gll_order;k0++)
        for(int k1=0;k0+k1   <=gll_order;k1++)
        for(int k2=0;k0+k1+k2<=gll_order;k2++){
          if(k0+k1+k2<=cheb_order){
            cheb_out[indx_out]=buff[indx_in];
            indx_out++;
          }
          indx_in++;
        }
      }
    }

    // Set avg to zero for Periodic boundary condition
    std::vector<pvfmm::Vector<double> > cheb_rho(node_cnt); //Copy vorticity.
    if(bndry==pvfmm::Periodic){
      std::vector<double> avg_rho(mykernel->ker_dim[0],0);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          avg_rho[k]+=node->ChebData()[k*n_cheb_coeff]*pow(0.5,node->Depth()*3);
      }
      std::vector<double> glb_avg_rho(mykernel->ker_dim[0],0);
      MPI_Allreduce(&avg_rho[0], &glb_avg_rho[0], mykernel->ker_dim[0], MPI_DOUBLE, MPI_SUM, fmm_data->comm);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          node->ChebData()[k*n_cheb_coeff]-=glb_avg_rho[k];
        cheb_rho[j]=node->ChebData();
      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    //Check Tree.
    pvfmm::Profile::Tic("Check Tree",&fmm_data->comm,true,1);
    mytree.CheckTree();
    pvfmm::Profile::Toc();
#endif

    //FMM Evaluate.
    pvfmm::Profile::Tic("FMM Eval",&fmm_data->comm,true,1);
    mytree.SetupFMM(&fmm_mat);
    mytree.RunFMM();
    pvfmm::Profile::Toc();

    //Copy FMM output to tree Data.
    mytree.Copy_FMMOutput();

    //Set average (grad phi) to zero. (Optional)
    if(bndry==pvfmm::Periodic){
      std::vector<double> avg_gradphi(mykernel->ker_dim[1],0);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          avg_gradphi[k]+=node->ChebData()[k*n_cheb_coeff]*pow(0.5,node->Depth()*3);
      }
      std::vector<double> glb_avg_gradphi(mykernel->ker_dim[1],0);
      MPI_Allreduce(&avg_gradphi[0], &glb_avg_gradphi[0], mykernel->ker_dim[1], MPI_DOUBLE, MPI_SUM, fmm_data->comm);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          node->ChebData()[k*n_cheb_coeff]-=glb_avg_gradphi[k];
      }
    }

#ifndef NDEBUG
    // Confirm that 2:1 balance did not change the number of tree nodes.
    size_t new_cnt=0;
    node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    while(node!=NULL){
      if(node->IsLeaf() && !node->IsGhost()) new_cnt++;
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }
    assert(node_cnt==new_cnt);
#endif

    // Construct GLL data from FMM output.
    pvfmm::Profile::Tic("Output",&fmm_data->comm,true,1);
    std::vector<double>& gll_nodes=*((std::vector<double>*)fmm_data->gll_nodes);
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      pvfmm::Vector<double> buff;
      pvfmm::Vector<double> buff1;
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        pvfmm::Vector<double> gll_data((gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[1],&node_gll_data[j][0],false);
        cheb_eval(tree_nodes[j]->ChebData(), cheb_order, gll_nodes, gll_nodes, gll_nodes, gll_data);
      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    //Compute |rho - div grad phi|_inf
    double resid_err=0;
    pvfmm::Profile::Tic("Divergence",&fmm_data->comm,true,1);
    pvfmm::Vector<double> buff0(n_cheb_coeff*mykernel->ker_dim[0]);
    pvfmm::Vector<double> buff1((gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[0]);
    for(size_t j=0;j<node_cnt;j++){
      tree_nodes[j]->Divergence();
      for(size_t k=0;k<cheb_rho[j].Dim();k++)
        buff0[k]=tree_nodes[j]->ChebData()[k]+cheb_rho[j][k];
      cheb_eval(buff0, cheb_order, gll_nodes, gll_nodes, gll_nodes, buff1);
      for(size_t k=0;k<buff1.Dim();k++)
        if(resid_err<fabs(buff1[k])) resid_err=fabs(buff1[k]);
    }
    pvfmm::Profile::Toc();

    //Display norms.
    double glb_max_resid_err=0;
    MPI_Allreduce(&resid_err, &glb_max_resid_err, 1, MPI_DOUBLE, MPI_MAX, fmm_data->comm);
    int myrank;
    MPI_Comm_rank(fmm_data->comm, &myrank);
    if(!myrank) std::cout<<"[fmm] |rho - div grad phi|_inf = "<<glb_max_resid_err<<'\n';
#endif

#ifndef NDEBUG
    pvfmm::Profile::print(&fmm_data->comm);
#endif
  }


  void gll_div(FMM_GLL_t* fmm_data, size_t node_cnt, double* node_coord, unsigned char* node_depth, double** node_gll_data){
    int myrank;
    MPI_Comm_rank(fmm_data->comm, &myrank);

    int dim=3;
    int gll_order=fmm_data->gll_order;
    int cheb_order=fmm_data->cheb_order;
    pvfmm::Kernel<double>* mykernel=((pvfmm::Kernel<double>*)fmm_data->kernel_biotsavart);

    // Find out number of OMP thereads.
    int omp_p=omp_get_max_threads();

    //Compute divergence.
    pvfmm::Profile::Tic("Div",&fmm_data->comm,true,1);
    size_t n_gll_coeff=(gll_order+1)*(gll_order+2)*(gll_order+3)/6;
    size_t n_cheb_coeff=(cheb_order+1)*(cheb_order+2)*(cheb_order+3)/6;
    std::vector<double>& gll_nodes=*((std::vector<double>*)fmm_data->gll_nodes);
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      pvfmm::Vector<double> buff(n_gll_coeff*mykernel->ker_dim[0]);
      pvfmm::Vector<double> cheb_coeff(n_cheb_coeff*mykernel->ker_dim[0]);
      pvfmm::Vector<double> cheb_div_coeff(n_cheb_coeff*mykernel->ker_dim[0]/dim);
      for(size_t j=start;j<end;j++){
        //Build Chebyshev approximation.
        pvfmm::gll2cheb<double,double>(&(node_gll_data[j][0]), gll_order, mykernel->ker_dim[0], &(buff[0]));

        //Truncate higher order terms.
        int indx_in=0;
        int indx_out=0;
        for(int k_=0;k_<mykernel->ker_dim[0];k_++)
        for(int k0=0;k0      <=gll_order;k0++)
        for(int k1=0;k0+k1   <=gll_order;k1++)
        for(int k2=0;k0+k1+k2<=gll_order;k2++){
          if(k0+k1+k2<=cheb_order){
            cheb_coeff[indx_out]=buff[indx_in];
            indx_out++;
          }
          indx_in++;
        }

        //Compute Divergence.
        double scale=(1<<node_depth[j]);
        for(int k=0;k<mykernel->ker_dim[0];k=k+dim)
          pvfmm::cheb_div<double>(&cheb_coeff[n_cheb_coeff*k], cheb_order, &cheb_div_coeff[n_cheb_coeff*(k/dim)]);
        for(size_t i=0;i<cheb_div_coeff.Dim();i++) cheb_div_coeff[i]*=scale;
        pvfmm::Vector<double> gll_data((gll_order+1)*(gll_order+1)*(gll_order+1)*1,&node_gll_data[j][0],false);
        cheb_eval(cheb_div_coeff, cheb_order, gll_nodes, gll_nodes, gll_nodes, gll_data);

      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    //Maximum divergence.
    double w_div=0;
    for(size_t j=0;j<node_cnt;j++){
      pvfmm::Vector<double> gll_data((gll_order+1)*(gll_order+1)*(gll_order+1)*1,&node_gll_data[j][0],false);
      for(size_t k=0;k<gll_data.Dim();k++)
        if(w_div<fabs(gll_data[k])) w_div=fabs(gll_data[k]);
    }

    double glb_w_div=0;
    MPI_Allreduce(&w_div, &glb_w_div, 1, MPI_DOUBLE, MPI_MAX, fmm_data->comm);
    if(!myrank) std::cout<<"[fmm] |div w|_inf      = "<<glb_w_div<<'\n';
#endif
  }

  void gll_divfree(FMM_GLL_t* fmm_data, size_t node_cnt, double* node_coord, unsigned char* node_depth, double** node_gll_data){
    int gll_order=fmm_data->gll_order;
    int cheb_order=fmm_data->cheb_order;
    FMM_Mat_t& fmm_mat=*((FMM_Mat_t*)fmm_data->fmm_mat_biotsavart);
    FMM_Tree_t& mytree=*((FMM_Tree_t*)fmm_data->tree_biotsavart);
    pvfmm::Kernel<double>* mykernel=((pvfmm::Kernel<double>*)fmm_data->kernel_biotsavart);
    pvfmm::BoundaryType bndry=pvfmm::Periodic;//pvfmm::FreeSpace;

    // Find out number of OMP thereads.
    int omp_p=omp_get_max_threads();

    pvfmm::Profile::Tic("Construct Tree",&fmm_data->comm,true,1);
    //Construct pvfmm::MortonId from node_coord and node_depth.
    std::vector<pvfmm::MortonId> nodes(node_cnt);
    double s=0.25/(1UL<<MAX_DEPTH);
    for(size_t i=0;i<node_cnt;i++)
      nodes[i]=pvfmm::MortonId(node_coord[i*3+0]+s,node_coord[i*3+1]+s,node_coord[i*3+2]+s,node_depth[i]);

    // Construct tree from morton id.
    size_t node_iter=0;
    std::vector<FMMNode_t*> tree_nodes;
    FMMNode_t* node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    while(node!=NULL && node_iter<node_cnt){
      pvfmm::MortonId mid=node->GetMortonId();
      if(mid.isAncestor(nodes[node_iter])){
        node->SetGhost(false);
        node->Subdivide();
      }else{
        node->Truncate();
        node->SetGhost(mid!=nodes[node_iter]);
        if(mid==nodes[node_iter]){
          tree_nodes.push_back(node);
          node_iter++;
        }
      }
      node->DataDOF()=mykernel->ker_dim[0];
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }
    assert(node_iter==node_cnt);
    while(node!=NULL){
      node->Truncate();
      node->SetGhost(true);
      node->DataDOF()=mykernel->ker_dim[0];
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }

    // Add GLL data
    size_t n_gll_coeff=(gll_order+1)*(gll_order+2)*(gll_order+3)/6;
    size_t n_cheb_coeff=(cheb_order+1)*(cheb_order+2)*(cheb_order+3)/6;
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      pvfmm::Vector<double> buff;
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        FMMNode_t* node=tree_nodes[j];
        node->DataDOF()=mykernel->ker_dim[0];
        buff.Resize(n_gll_coeff*node->DataDOF());
        pvfmm::gll2cheb<double,double>(&(node_gll_data[j][0]), gll_order, node->DataDOF(), &(buff[0]));

        node->ChebData().Resize(n_cheb_coeff*node->DataDOF());
        double* cheb_out=&(node->ChebData()[0]);
        int indx_in=0;
        int indx_out=0;
        for(int k_=0;k_<node->DataDOF();k_++)
        for(int k0=0;k0      <=gll_order;k0++)
        for(int k1=0;k0+k1   <=gll_order;k1++)
        for(int k2=0;k0+k1+k2<=gll_order;k2++){
          if(k0+k1+k2<=cheb_order){
            cheb_out[indx_out]=buff[indx_in];
            indx_out++;
          }
          indx_in++;
        }
      }
    }

    // Set avg to zero for Periodic boundary condition
    if(bndry==pvfmm::Periodic){
      std::vector<double> avg_w(mykernel->ker_dim[0],0);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          avg_w[k]+=node->ChebData()[k*n_cheb_coeff]*pow(0.5,node->Depth()*3);
      }
      std::vector<double> glb_avg_w(mykernel->ker_dim[0],0);
      MPI_Allreduce(&avg_w[0], &glb_avg_w[0], mykernel->ker_dim[0], MPI_DOUBLE, MPI_SUM, fmm_data->comm);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          node->ChebData()[k*n_cheb_coeff]-=glb_avg_w[k];
      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    //Check Tree.
    pvfmm::Profile::Tic("Check Tree",&fmm_data->comm,true,1);
    mytree.CheckTree();
    pvfmm::Profile::Toc();
#endif

    //FMM Evaluate.
    pvfmm::Profile::Tic("FMM Eval",&fmm_data->comm,true,1);
    mytree.SetupFMM(&fmm_mat);
    mytree.RunFMM();
    pvfmm::Profile::Toc();

    //Copy FMM output to tree Data.
    mytree.Copy_FMMOutput();

    //Set average velocity to zero.
    if(bndry==pvfmm::Periodic){
      std::vector<double> avg_v(mykernel->ker_dim[1],0);
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          avg_v[k]+=node->ChebData()[k*n_cheb_coeff]*pow(0.5,node->Depth()*3);
      }
      std::vector<double> glb_avg_v(mykernel->ker_dim[1],0);
      MPI_Allreduce(&avg_v[0], &glb_avg_v[0], mykernel->ker_dim[1], MPI_DOUBLE, MPI_SUM, fmm_data->comm);
      //std::cout<<glb_avg_v[0]<<' '<<glb_avg_v[1]<<' '<<glb_avg_v[2]<<'\n';
      for(size_t j=0;j<node_cnt;j++){
        FMMNode_t* node=tree_nodes[j];
        for(int k=0;k<node->DataDOF();k++)
          node->ChebData()[k*n_cheb_coeff]-=glb_avg_v[k];
      }
    }

#ifndef NDEBUG
    // Confirm that 2:1 balance did not change the number of tree nodes.
    size_t new_cnt=0;
    node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    while(node!=NULL){
      if(node->IsLeaf() && !node->IsGhost()) new_cnt++;
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }
    assert(node_cnt==new_cnt);
#endif

    // Construct GLL data from FMM output.
    pvfmm::Profile::Tic("Output",&fmm_data->comm,true,1);
    std::vector<double>& gll_nodes=*((std::vector<double>*)fmm_data->gll_nodes);
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        tree_nodes[j]->Curl();
        pvfmm::Vector<double> gll_data((gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[1],&node_gll_data[j][0],false);
        cheb_eval(tree_nodes[j]->ChebData(), cheb_order+1, gll_nodes, gll_nodes, gll_nodes, gll_data);
      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    pvfmm::Profile::print(&fmm_data->comm);
#endif
  }

  void gll_filter(FMM_GLL_t* fmm_data, int cheb_order_, size_t node_cnt, double** node_gll_data, double* err){
    int gll_order=fmm_data->gll_order;
    int cheb_order=cheb_order_-1;
    assert( gll_order>1);
    assert(cheb_order>1);

    size_t n_gll     =( gll_order+1)*( gll_order+1)*( gll_order+1);
    size_t n_cheb_in =( gll_order+1)*( gll_order+2)*( gll_order+3)/6;
    size_t n_cheb_out=(cheb_order+1)*(cheb_order+2)*(cheb_order+3)/6;
    std::vector<double>& gll_nodes=*((std::vector<double>*)fmm_data->gll_nodes);
    //double max_err=0, min_rate=1e+10;

    // Find out number of OMP thereads.
    int omp_p=omp_get_max_threads();

    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      pvfmm::Vector<double> cheb_in (n_cheb_in );
      pvfmm::Vector<double> cheb_out(n_cheb_out);
      pvfmm::Vector<double> error(gll_order+1);

      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        pvfmm::gll2cheb<double,double>(&(node_gll_data[j][0]), gll_order, 1, &(cheb_in[0]));

        error.SetZero();
        int indx_in=0;
        int indx_out=0;
        for(int k0=0;k0      <=gll_order;k0++)
        for(int k1=0;k0+k1   <=gll_order;k1++)
        for(int k2=0;k0+k1+k2<=gll_order;k2++){
          error[k0+k1+k2]+=fabs(cheb_in[indx_in]);
          if(k0+k1+k2<=cheb_order){
            cheb_out[indx_out]=cheb_in[indx_in];
            indx_out++;
          }
          indx_in++;
        }

        // Compute error.
        {
          // Fit: y ~ A exp(B.k), using last k0 error values.
          int k0=(cheb_order+1<4?cheb_order+1:4);
          double A,B;
          double k_=0, k2_=0;
          double lny_=0, klny_=0;
          for(int k=0;k<k0;k++){
            k_+=((double)k);
            k2_+=((double)k*k);
            lny_+=log(error[cheb_order-k]+1e-20);
            klny_+=k*log(error[cheb_order-k]+1e-20);
          }
          B=-(k0*klny_-k_*lny_)/(k0*k2_-k_*k_);

          double ye_=0, e2_=0;
          for(int k=0;k<k0;k++){
            ye_+=error[cheb_order-k]*exp(B*k);
            e2_+=exp(2*B*k);
          }
          A=(ye_/e2_);

          // err+= integ_{cheb_order}^{inf}
          err[j]+=error[cheb_order];
          err[j]+=(-B>1e-3?-A/B:A);

          //if(max_err<err[j]) max_err=err[j];
          //if(-B>1e-3 && -B<min_rate) min_rate=-B;

          //for(int k=0;k<k0;k++)
          //  std::cout<<error[cheb_order-k-1]<<' ';
          //std::cout<<'\n';
          //std::cout<<A<<' '<<B<<'\n';
        }

        pvfmm::Vector<double> gll_data(n_gll,&node_gll_data[j][0],false);
        cheb_eval(cheb_out, cheb_order, gll_nodes, gll_nodes, gll_nodes, gll_data);
      }
    }
    //std::cout<<max_err<<' '<<min_rate<<'\n';
  }

  void gll_interpolate(FMM_GLL_t* fmm_data, size_t node_cnt, double* node_coord, unsigned char* node_depth, double** node_gll_data){
    int gll_order=fmm_data->gll_order;
    int cheb_order=fmm_data->cheb_order;
    FMM_Tree_t& mytree=*((FMM_Tree_t*)fmm_data->tree_biotsavart);
    pvfmm::Kernel<double>* mykernel=((pvfmm::Kernel<double>*)fmm_data->kernel_biotsavart);
    pvfmm::BoundaryType bndry=pvfmm::Periodic;//pvfmm::FreeSpace;

    // Find out number of OMP thereads.
    int omp_p=omp_get_max_threads();

    pvfmm::Profile::Tic("Construct Tree",&fmm_data->comm,true,1);
    //Construct pvfmm::MortonId from node_coord and node_depth.
    std::vector<pvfmm::MortonId> nodes(node_cnt);
    double s=0.25/(1UL<<MAX_DEPTH);
    for(size_t i=0;i<node_cnt;i++)
      nodes[i]=pvfmm::MortonId(node_coord[i*3+0]+s,node_coord[i*3+1]+s,node_coord[i*3+2]+s,node_depth[i]);

    // Construct tree from morton id.
    size_t node_iter=0;
    std::vector<FMMNode_t*> tree_nodes;
    FMMNode_t* node=static_cast<FMMNode_t*>(mytree.PreorderFirst());
    while(node!=NULL && node_iter<node_cnt){
      pvfmm::MortonId mid=node->GetMortonId();
      node->DataDOF()=mykernel->ker_dim[0];
      if(mid.isAncestor(nodes[node_iter])){
        node->SetGhost(false);
        node->Subdivide();
      }else{
        node->Truncate();
        node->SetGhost(mid!=nodes[node_iter]);
        if(mid==nodes[node_iter]){
          tree_nodes.push_back(node);
          node_iter++;
        }
      }
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }
    assert(node_iter==node_cnt);
    while(node!=NULL){
      node->Truncate();
      node->SetGhost(true);
      node->DataDOF()=mykernel->ker_dim[0];
      node=static_cast<FMMNode_t*>(mytree.PreorderNxt(node));
    }

    // Add GLL data
    size_t n_gll_coeff=(gll_order+1)*(gll_order+2)*(gll_order+3)/6;
    size_t n_cheb_coeff=(cheb_order+1)*(cheb_order+2)*(cheb_order+3)/6;
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      pvfmm::Vector<double> buff;
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        FMMNode_t* node=tree_nodes[j];
        buff.Resize(n_gll_coeff*node->DataDOF());
        pvfmm::gll2cheb<double,double>(&(node_gll_data[j][0]), gll_order, node->DataDOF(), &(buff[0]));

        node->ChebData().Resize(n_cheb_coeff*node->DataDOF());
        double* cheb_out=&(node->ChebData()[0]);
        int indx_in=0;
        int indx_out=0;
        for(int k_=0;k_<node->DataDOF();k_++)
        for(int k0=0;k0      <=gll_order;k0++)
        for(int k1=0;k0+k1   <=gll_order;k1++)
        for(int k2=0;k0+k1+k2<=gll_order;k2++){
          if(k0+k1+k2<=cheb_order){
            cheb_out[indx_out]=buff[indx_in];
            indx_out++;
          }
          indx_in++;
        }
      }
    }

#ifndef NDEBUG
    //Check Tree.
    pvfmm::Profile::Tic("Check Tree",&fmm_data->comm,true,1);
    mytree.CheckTree();
    pvfmm::Profile::Toc();
#endif

    // Get neighbour data.
    pvfmm::Profile::Tic("ConstructLET",&fmm_data->comm,true,1);
    mytree.ConstructLET(bndry);
    pvfmm::Profile::Toc();

    // Interpolate across nearby octants.
    pvfmm::Profile::Tic("Interpolate",&fmm_data->comm,true,1);
    double alpha=1.25;
    FMMNode_t* r_node=static_cast<FMMNode_t*>(mytree.RootNode());
    pvfmm::Matrix<double> cheb_data(node_cnt, n_cheb_coeff*mykernel->ker_dim[0]);
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      std::vector<double> x1(cheb_order+1);
      std::vector<double> cheb_pts1=pvfmm::cheb_nodes<double>(cheb_order+1,1);
      for(int k=0;k<=cheb_order;k++) x1[k]=(cheb_pts1[k]-0.5)*2.0/alpha;

      std::vector<double> x2(2*(cheb_order+1));
      std::vector<double> y2(2*(cheb_order+1));
      std::vector<double> z2(2*(cheb_order+1));
      std::vector<double> x2_(2*(cheb_order+1));
      std::vector<double> y2_(2*(cheb_order+1));
      std::vector<double> z2_(2*(cheb_order+1));
      std::vector<int> px(2*(cheb_order+1)), py(2*(cheb_order+1)), pz(2*(cheb_order+1));
      std::vector<double> cheb_pts2=pvfmm::cheb_nodes<double>(2*cheb_order,1);

      pvfmm::Vector<double> buff0 ( 2*(cheb_order+1)* 2*(cheb_order+1)   * 2*(cheb_order+1)      *mykernel->ker_dim[0]);
      pvfmm::Vector<double> buff0_( 2*(cheb_order+1)* 2*(cheb_order+1)   * 2*(cheb_order+1)      *mykernel->ker_dim[0]);
      pvfmm::Vector<double> buff1 ((2*(cheb_order+1)*(2*(cheb_order+1)+1)*(2*(cheb_order+1)+2)/6)*mykernel->ker_dim[0]);
      pvfmm::Vector<double> buff2( (cheb_order+1)*(cheb_order+1)*(cheb_order+1)   *mykernel->ker_dim[0]);
      pvfmm::Vector<double> buff3(((cheb_order+1)*(cheb_order+2)*(cheb_order+3)/6)*mykernel->ker_dim[0]);

      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        FMMNode_t* node=tree_nodes[j];
        double s=pow(0.5,node->Depth());
        double* coord=tree_nodes[j]->Coord();

        for(int k=0;k<2*(cheb_order+1);k++){
          x2[k]=(coord[0]+0.5*s)+s*(cheb_pts2[k]-0.5)*alpha+1.0;
          y2[k]=(coord[1]+0.5*s)+s*(cheb_pts2[k]-0.5)*alpha+1.0;
          z2[k]=(coord[2]+0.5*s)+s*(cheb_pts2[k]-0.5)*alpha+1.0;
          if(x2[k]>=1.0) x2[k]-=1.0; if(x2[k]>=1.0) x2[k]-=1.0;
          if(y2[k]>=1.0) y2[k]-=1.0; if(y2[k]>=1.0) y2[k]-=1.0;
          if(z2[k]>=1.0) z2[k]-=1.0; if(z2[k]>=1.0) z2[k]-=1.0;
        }
        x2_=x2; y2_=y2; z2_=z2;
        std::sort(x2.begin(),x2.end());
        std::sort(y2.begin(),y2.end());
        std::sort(z2.begin(),z2.end());
        for(int k0=0;k0<2*(cheb_order+1);k0++)
        for(int k1=0;k1<2*(cheb_order+1);k1++){
          if(x2[k0]==x2_[k1]) px[k1]=k0;
          if(y2[k0]==y2_[k1]) py[k1]=k0;
          if(z2[k0]==z2_[k1]) pz[k1]=k0;
        }
        r_node->ReadVal(x2, y2, z2, &buff0[0]);
        for(int l=0;l<mykernel->ker_dim[0];l++)
        for(int k0=0;k0<2*(cheb_order+1);k0++)
        for(int k1=0;k1<2*(cheb_order+1);k1++)
        for(int k2=0;k2<2*(cheb_order+1);k2++){
          buff0_[k2+(k1+(k0+l*2*(cheb_order+1))*2*(cheb_order+1))*2*(cheb_order+1)]=
            buff0[px[k2]+(py[k1]+(pz[k0]+l*2*(cheb_order+1))*2*(cheb_order+1))*2*(cheb_order+1)];
        }

        pvfmm::cheb_approx<double,double>(&buff0_[0], 2*cheb_order+1, mykernel->ker_dim[0], &buff1[0]);

        //TODO: Truncate?

        cheb_eval(buff1, 2*cheb_order+1, x1, x1, x1, buff2);
        pvfmm::cheb_approx<double,double>(&buff2[0],   cheb_order, mykernel->ker_dim[0], &buff3[0]);
        node->ChebData()=buff3;
      }
    }
    pvfmm::Profile::Toc();

    // Cheb2GLL.
    pvfmm::Profile::Tic("Cheb2GLL",&fmm_data->comm,true,1);
    std::vector<double>& gll_nodes=*((std::vector<double>*)fmm_data->gll_nodes);
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t start=i*node_cnt/omp_p;
      size_t end=(i+1)*node_cnt/omp_p;
      for(size_t j=start;j<end;j++){
        pvfmm::Vector<double> gll_data((gll_order+1)*(gll_order+1)*(gll_order+1)*mykernel->ker_dim[1],&node_gll_data[j][0],false);
        cheb_eval(tree_nodes[j]->ChebData(), cheb_order, gll_nodes, gll_nodes, gll_nodes, gll_data);
      }
    }
    pvfmm::Profile::Toc();

#ifndef NDEBUG
    pvfmm::Profile::print(&fmm_data->comm);
#endif
  }

}
