#include <mpi.h>
#include <pvfmm_common.hpp>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <stdio.h>

#include <profile.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <utils.hpp>

//////////////////////////////////////////////////////////////////////////////
// Test1: Laplace problem, Smooth Gaussian, Periodic Boundary
///////////////////////////////////////////////////////////////////////////////
template <class Real_t>
void fn_input_t1(Real_t* coord, int n, Real_t* out){ //Input function
  int dof=1;
  Real_t a=-160;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.0)*(c[2]-0.0);
      out[i*dof+0]=(2*a*r_2+3)*2*a*exp(a*r_2);
    }
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-1.0)*(c[2]-1.0);
      out[i*dof+0]+=(2*a*r_2+3)*2*a*exp(a*r_2);
    }
  }
}
template <class Real_t>
void fn_poten_t1(Real_t* coord, int n, Real_t* out){ //Output potential
  int dof=1;
  Real_t a=-160;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.0)*(c[2]-0.0);
      out[i*dof+0]=-exp(a*r_2);
    }
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-1.0)*(c[2]-1.0);
      out[i*dof+0]+=-exp(a*r_2);
    }
  }
}
template <class Real_t>
void fn_grad_t1(Real_t* coord, int n, Real_t* out){ //Output gradient
  int dof=1;
  Real_t a=-160;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.0)*(c[2]-0.0);
      out[i*dof+0]=-2*a*exp(a*r_2)*(c[0]-0.5);
      out[i*dof+1]=-2*a*exp(a*r_2)*(c[1]-0.5);
      out[i*dof+2]=-2*a*exp(a*r_2)*(c[2]-0.0);
    }
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-1.0)*(c[2]-1.0);
      out[i*dof+0]+=-2*a*exp(a*r_2)*(c[0]-0.5);
      out[i*dof+1]+=-2*a*exp(a*r_2)*(c[1]-0.5);
      out[i*dof+2]+=-2*a*exp(a*r_2)*(c[2]-1.0);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
// Test2: Laplace problem, Discontinuous Sphere, FreeSpace Boundary
///////////////////////////////////////////////////////////////////////////////
template <class Real_t>
void fn_input_t2(Real_t* coord, int n, Real_t* out){ //Input function
  int dof=1;
  Real_t R=0.1;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*3];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=(sqrt(r_2)<R?1:0);
    }
  }
}
template <class Real_t>
void fn_poten_t2(Real_t* coord, int n, Real_t* out){ //Output potential
  int dof=1;
  Real_t R=0.1;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*3];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=(sqrt(r_2)<R? (R*R-r_2)/6 + R*R/3 : pow(R,3)/(3*sqrt(r_2)) );
    }
  }
}
template <class Real_t>
void fn_grad_t2(Real_t* coord, int n, Real_t* out){ //Output gradient
  int dof=3;
  Real_t R=0.1;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*3];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=(sqrt(r_2)<R? -r_2/3 : -pow(R,3)/(3*sqrt(r_2)) )*(c[0]-0.5)/r_2;
      out[i*dof+1]=(sqrt(r_2)<R? -r_2/3 : -pow(R,3)/(3*sqrt(r_2)) )*(c[1]-0.5)/r_2;
      out[i*dof+2]=(sqrt(r_2)<R? -r_2/3 : -pow(R,3)/(3*sqrt(r_2)) )*(c[2]-0.5)/r_2;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
// Test3: Stokes problem, Smooth Gaussian, FreeSpace Boundary
///////////////////////////////////////////////////////////////////////////////
template <class Real_t>
void fn_input_t3(Real_t* coord, int n, Real_t* out){ //Input function
  int dof=3;
  Real_t L=125;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]= 0;
      out[i*dof+1]= 4*L*L*(c[2]-0.5)*(5-2*L*r_2)*exp(-L*r_2);
      out[i*dof+2]=-4*L*L*(c[1]-0.5)*(5-2*L*r_2)*exp(-L*r_2);
    }
  }
}
template <class Real_t>
void fn_poten_t3(Real_t* coord, int n, Real_t* out){ //Output potential
  int dof=3;
  Real_t L=125;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]= 0;
      out[i*dof+1]= 2*L*(c[2]-0.5)*exp(-L*r_2);
      out[i*dof+2]=-2*L*(c[1]-0.5)*exp(-L*r_2);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
// Test4: Biot-Savart problem, Smooth Gaussian, FreeSpace Boundary
///////////////////////////////////////////////////////////////////////////////
template <class Real_t>
void fn_input_t4(Real_t* coord, int n, Real_t* out){ //Input function
  int dof=3;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=0;
      out[i*dof+1]=0;
      out[i*dof+2]=0;
    }
  }
}
template <class Real_t>
void fn_poten_t4(Real_t* coord, int n, Real_t* out){ //Output potential
  int dof=3;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=0;
      out[i*dof+1]=0;
      out[i*dof+2]=0;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
// Test5: Helmholtz problem, Smooth Gaussian, FreeSpace Boundary
///////////////////////////////////////////////////////////////////////////////
template <class Real_t>
void fn_input_t5(Real_t* coord, int n, Real_t* out){
  int dof=2;
  Real_t a=-160;
  Real_t mu=(20.0*M_PI);
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*3];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=((2*a*r_2+3)*2*a*exp(a*r_2)+mu*mu*exp(a*r_2))/4.0/M_PI;
      out[i*dof+1]=0;
    }
  }
}
template <class Real_t>
void fn_poten_t5(Real_t* coord, int n, Real_t* out){
  int dof=2;
  Real_t a=-160;
  for(int i=0;i<n;i++){
    Real_t* c=&coord[i*3];
    {
      Real_t r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      out[i*dof+0]=-exp(a*r_2);
      out[i*dof+1]=0;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////

template <class Real_t>
void fmm_test(int test_case, size_t N, bool unif, int mult_order, int cheb_deg, int depth, bool adap, Real_t tol, MPI_Comm comm){
  typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<Real_t> > FMMNode_t;
  typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
  typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;

  void (*fn_input_)(Real_t* , int , Real_t*)=NULL;
  void (*fn_poten_)(Real_t* , int , Real_t*)=NULL;
  void (*fn_grad_)(Real_t* , int , Real_t*)=NULL;
  const pvfmm::Kernel<Real_t>* mykernel=NULL;
  const pvfmm::Kernel<Real_t>* mykernel_grad=NULL;;
  pvfmm::BoundaryType bndry;

  switch (test_case){
    case 1:
      fn_input_=fn_input_t1<Real_t>;
      fn_poten_=fn_poten_t1<Real_t>;
      fn_grad_ =fn_grad_t1<Real_t>;
      mykernel     =pvfmm::LaplaceKernel<Real_t>::potn_ker;
      //mykernel_grad=pvfmm::LaplaceKernel<Real_t>::grad_ker;
      bndry=pvfmm::Periodic;
      //bndry=pvfmm::FreeSpace;
      break;
    case 2:
      fn_input_=fn_input_t2<Real_t>;
      fn_poten_=fn_poten_t2<Real_t>;
      fn_grad_ =fn_grad_t2<Real_t>;
      mykernel     =pvfmm::LaplaceKernel<Real_t>::potn_ker;
      //mykernel_grad=pvfmm::LaplaceKernel<Real_t>::grad_ker;
      bndry=pvfmm::FreeSpace;
      break;
    case 3:
      fn_input_=fn_input_t3<Real_t>;
      fn_poten_=fn_poten_t3<Real_t>;
      mykernel     =&pvfmm::ker_stokes_vel;
      //mykernel_grad=&pvfmm::ker_stokes_grad;
      bndry=pvfmm::FreeSpace;
      break;
    case 4:
      fn_input_=fn_input_t4<Real_t>;
      fn_poten_=fn_poten_t4<Real_t>;
      mykernel     =&pvfmm::ker_biot_savart;
      //mykernel_grad=&pvfmm::ker_biot_savart_grad;
      bndry=pvfmm::FreeSpace;
      break;
    case 5:
      fn_input_=fn_input_t5<Real_t>;
      fn_poten_=fn_poten_t5<Real_t>;
      mykernel     =&pvfmm::ker_helmholtz;
      //mykernel_grad=&pvfmm::ker_helmholtz_grad;
      bndry=pvfmm::FreeSpace;
      break;
    default:
      fn_input_=NULL;
      fn_poten_=NULL;
      fn_grad_ =NULL;
      break;
  }

  // Find out number of OMP thereads.
  int omp_p=omp_get_max_threads();

  // Find out my identity in the default communicator
  int myrank, p;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm,&p);

  //Various parameters.
  typename FMMNode_t::NodeData tree_data;
  tree_data.dim=COORD_DIM;
  tree_data.max_depth=depth;
  tree_data.cheb_deg=cheb_deg;

  //Set input function pointer
  tree_data.input_fn=fn_input_;
  tree_data.data_dof=mykernel->ker_dim[0];
  tree_data.tol=tol;

  //Set source coordinates.
  std::vector<Real_t> pt_coord;
  if(unif) pt_coord=point_distrib<Real_t>(UnifGrid,N,comm);
  else pt_coord=point_distrib<Real_t>(RandElps,N,comm); //RandElps, RandGaus
  tree_data.max_pts=1; // Points per octant.
  tree_data.pt_coord=pt_coord;

  //Print various parameters.
  if(!myrank){
    std::cout<<std::setprecision(2)<<std::scientific;
    std::cout<<"Number of MPI processes: "<<p<<'\n';
    std::cout<<"Number of OpenMP threads: "<<omp_p<<'\n';
    std::cout<<"Order of multipole expansions: "<<mult_order<<'\n';
    std::cout<<"Order of Chebyshev polynomials: "<<tree_data.cheb_deg<<'\n';
    std::cout<<"FMM Kernel name: "<<mykernel->ker_name<<'\n';
    std::cout<<"Number of point samples: "<<N<<'\n';
    std::cout<<"Uniform distribution: "<<(unif?"true":"false")<<'\n';
    std::cout<<"Maximum points per octant: "<<tree_data.max_pts<<'\n';
    std::cout<<"Chebyshev Tolerance: "<<tree_data.tol<<'\n';
    std::cout<<"Maximum Tree Depth: "<<depth<<'\n';
    std::cout<<"BoundaryType: "<<(bndry==pvfmm::Periodic?"Periodic":"FreeSpace")<<'\n';
  }

  //Initialize FMM_Mat.
  FMM_Mat_t* fmm_mat=NULL;
  FMM_Mat_t* fmm_mat_grad=NULL;
  {
    fmm_mat=new FMM_Mat_t;
    fmm_mat->Initialize(mult_order,tree_data.cheb_deg,comm,mykernel);
  }
  if(mykernel_grad!=NULL){
    fmm_mat_grad=new FMM_Mat_t;
    fmm_mat_grad->Initialize(mult_order,tree_data.cheb_deg,comm,mykernel_grad,mykernel);
  }

  pvfmm::Profile::Tic("TreeSetup",&comm,true,1);
  {
    FMM_Tree_t* tree=new FMM_Tree_t(comm);
    tree->Initialize(&tree_data);

    tree->InitFMM_Tree(adap,bndry); //Adaptive refinement.

    pt_coord.clear();
    FMMNode_t* node=static_cast<FMMNode_t*>(tree->PreorderFirst());
    while(node!=NULL){
      if(node->IsLeaf() && !node->IsGhost()){
        Real_t* c=node->Coord();
        Real_t s=pow(0.5,node->Depth()+1);
        pt_coord.push_back(c[0]+s);
        pt_coord.push_back(c[1]+s);
        pt_coord.push_back(c[2]+s);
      }
      node=static_cast<FMMNode_t*>(tree->PreorderNxt(node));
    }
    delete tree;
    tree_data.pt_coord=pt_coord;
  }
  pvfmm::Profile::Toc();

  //Create Tree and initialize with input data.
  FMM_Tree_t* tree=new FMM_Tree_t(comm);
  tree->Initialize(&tree_data);

  //Initialize FMM Tree
  tree->InitFMM_Tree(false,bndry);

  { //Output max tree depth.
    std::vector<size_t> all_nodes(MAX_DEPTH+1,0);
    std::vector<size_t> leaf_nodes(MAX_DEPTH+1,0);
    std::vector<FMMNode_t*>& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMMNode_t* n=nodes[i];
      if(!n->IsGhost()) all_nodes[n->Depth()]++;
      if(!n->IsGhost() && n->IsLeaf()) leaf_nodes[n->Depth()]++;
    }

    if(!myrank) std::cout<<"All  Nodes: ";
    for(int i=0;i<MAX_DEPTH;i++){
      int local_size=all_nodes[i];
      int global_size;
      MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
      if(!myrank) std::cout<<global_size<<' ';
    }
    if(!myrank) std::cout<<'\n';

    if(!myrank) std::cout<<"Leaf Nodes: ";
    for(int i=0;i<MAX_DEPTH;i++){
      int local_size=leaf_nodes[i];
      int global_size;
      MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
      if(!myrank) std::cout<<global_size<<' ';
    }
    if(!myrank) std::cout<<'\n';
  }

  //Find error in Chebyshev approximation.
  CheckChebOutput<FMM_Tree_t>(tree, (typename TestFn<Real_t>::Fn_t) fn_input_, mykernel->ker_dim[0], std::string("Input"));

  //FMM Evaluate.
  tree->SetupFMM(fmm_mat);
  tree->RunFMM();

  //Re-run FMM
  tree->ClearFMMData();
  tree->RunFMM();

  //Re-run FMM
  tree->ClearFMMData();
  tree->RunFMM();

  tree->Copy_FMMOutput(); //Copy FMM output to tree Data.

  //Check Tree.
  #ifndef NDEBUG
  pvfmm::Profile::Tic("CheckTree",&comm,true,1);
  tree->CheckTree();
  pvfmm::Profile::Toc();
  #endif

  //Find error in FMM output.
  CheckChebOutput<FMM_Tree_t>(tree, (typename TestFn<Real_t>::Fn_t) fn_poten_, mykernel->ker_dim[1], std::string("Output"));

  //Write2File
  tree->Write2File("result/output",0);

  if(fn_grad_!=NULL){ //Compute gradient.
    pvfmm::Profile::Tic("FMM_Eval(Grad)",&comm,true,1);
    if(mykernel_grad!=NULL){
      tree->SetupFMM(fmm_mat_grad);
      tree->DownwardPass();
      tree->Copy_FMMOutput(); //Copy FMM output to tree Data.
    }else{
      std::vector<FMMNode_t*> nlist=tree->GetNodeList();
      #pragma omp parallel for
      for(size_t i=0;i<nlist.size();i++) nlist[i]->Gradient();
    }
    pvfmm::Profile::Toc();

    //Find error in FMM output (gradient).
    CheckChebOutput<FMM_Tree_t>(tree, (typename TestFn<Real_t>::Fn_t) fn_grad_, mykernel->ker_dim[1]*COORD_DIM, std::string("OutputGrad"));
  }

  //Delete matrices.
  if(fmm_mat     ) delete fmm_mat     ;
  if(fmm_mat_grad) delete fmm_mat_grad;

  //Delete the tree.
  delete tree;
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  MPI_Comm comm=MPI_COMM_WORLD;

  // Read command line options.
  commandline_option_start(argc, argv);
  omp_set_num_threads( atoi(commandline_option(argc, argv,  "-omp",     "1", false, "-omp  <int> = (1)    : Number of OpenMP threads."          )));
  size_t   N=(size_t)strtod(commandline_option(argc, argv,    "-N",     "1",  true, "-N    <int>          : Number of point sources."           ),NULL);
  bool  unif=              (commandline_option(argc, argv, "-unif",    NULL, false, "-unif                : Uniform point distribution."        )!=NULL);
  int      m=       strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  int      q=       strtoul(commandline_option(argc, argv,    "-q",    "14", false, "-q    <int> = (14)   : Chebyshev order (+ve integer)."     ),NULL,10);
  int      d=       strtoul(commandline_option(argc, argv,    "-d",    "15", false, "-d    <int> = (15)   : Maximum tree depth."                ),NULL,10);
  double tol=        strtod(commandline_option(argc, argv,  "-tol",  "1e-5", false, "-tol <real> = (1e-5) : Tolerance for adaptive refinement." ),NULL);
  bool  adap=              (commandline_option(argc, argv, "-adap",    NULL, false, "-adap                : Adaptive tree refinement."          )!=NULL);
  int   test=       strtoul(commandline_option(argc, argv, "-test",     "1", false,
       "-test <int> = (1)    : 1) Laplace, Smooth Gaussian, Periodic Boundary\n\
                               2) Laplace, Discontinuous Sphere, FreeSpace Boundary\n\
                               3) Stokes, Smooth Gaussian, FreeSpace Boundary\n\
                               4) Biot-Savart, Smooth Gaussian, FreeSpace Boundary\n\
                               5) Helmholtz, Smooth Gaussian, FreeSpace Boundary"),NULL,10);
  commandline_option_end(argc, argv);

  // Run FMM with above options.
  pvfmm::Profile::Tic("FMM_Test",&comm,true);
  fmm_test<double>(test, N,unif, m,q, d, adap,tol, comm);
  pvfmm::Profile::Toc();

  //Output Profiling results.
  pvfmm::Profile::print(&comm);

  // Shut down MPI
  MPI_Finalize();
  return 0;
}

