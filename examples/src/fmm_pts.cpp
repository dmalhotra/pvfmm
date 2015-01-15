#include <mpi.h>
#include <pvfmm_common.hpp>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <stdio.h>

#include <profile.hpp>
#include <fmm_pts.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <utils.hpp>

template <class Real_t>
void fmm_test(int ker, size_t N, size_t M, Real_t b, int dist, int mult_order, int depth, MPI_Comm comm){
  typedef pvfmm::FMM_Node<pvfmm::MPI_Node<Real_t> > FMMNode_t;
  typedef pvfmm::FMM_Pts<FMMNode_t> FMM_Mat_t;
  typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;

  //Set kernel.
  const pvfmm::Kernel<Real_t>* mykernel=NULL;
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

  switch (ker){
    case 1:
      mykernel     =&pvfmm::LaplaceKernel<Real_t>::potential();
      break;
    case 2:
      mykernel     =&pvfmm::LaplaceKernel<Real_t>::gradient();
      break;
    case 3:
      mykernel     =&pvfmm::HelmholtzKernel<Real_t>::potential();
      break;
    default:
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
  tree_data.max_pts=M; // Points per octant.

  //Set source coordinates and values.
  std::vector<Real_t> src_coord, src_value;
  src_coord=point_distrib<Real_t>((dist==0?UnifGrid:(dist==1?RandSphr:RandElps)),N,comm);
  for(size_t i=0;i<src_coord.size();i++) src_coord[i]*=b;
  for(size_t i=0;i<src_coord.size()*mykernel->ker_dim[0]/COORD_DIM;i++) src_value.push_back(drand48());
  tree_data.pt_coord=src_coord;
  tree_data.src_coord=src_coord;
  tree_data.src_value=src_value;

  //Set target coordinates.
  tree_data.trg_coord=tree_data.src_coord;

  //Print various parameters.
  if(!myrank){
    std::cout<<std::setprecision(2)<<std::scientific;
    std::cout<<"Number of MPI processes: "<<p<<'\n';
    std::cout<<"Number of OpenMP threads: "<<omp_p<<'\n';
    std::cout<<"Order of multipole expansions: "<<mult_order<<'\n';
    std::cout<<"FMM Kernel name: "<<mykernel->ker_name<<'\n';
    std::cout<<"Number of point samples: "<<N<<'\n';
    std::cout<<"Point distribution: "<<(dist==0?"Unif":(dist==1?"Sphere":"Ellipse"))<<'\n';
    std::cout<<"Maximum points per octant: "<<tree_data.max_pts<<'\n';
    std::cout<<"Maximum Tree Depth: "<<depth<<'\n';
    std::cout<<"BoundaryType: "<<(bndry==pvfmm::Periodic?"Periodic":"FreeSpace")<<'\n';
  }

  //Initialize FMM_Mat.
  FMM_Mat_t* fmm_mat=new FMM_Mat_t;
  fmm_mat->Initialize(mult_order,comm,mykernel);

  //Create Tree and initialize with input data.
  FMM_Tree_t* tree=new FMM_Tree_t(comm);
  tree->Initialize(&tree_data);

  //Initialize FMM Tree
  tree->InitFMM_Tree(false,bndry);

  { //Output max tree depth.
    long nleaf=0, maxdepth=0;
    std::vector<size_t> all_nodes(MAX_DEPTH+1,0);
    std::vector<size_t> leaf_nodes(MAX_DEPTH+1,0);
    std::vector<FMMNode_t*>& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMMNode_t* n=nodes[i];
      if(!n->IsGhost()) all_nodes[n->Depth()]++;
      if(!n->IsGhost() && n->IsLeaf()){
        leaf_nodes[n->Depth()]++;
        if(maxdepth<n->Depth()) maxdepth=n->Depth();
        nleaf++;
      }
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

    long nleaf_glb=0, maxdepth_glb=0;
    { // MPI_Reduce
      MPI_Allreduce(&nleaf, &nleaf_glb, 1, MPI_INT, MPI_SUM, comm);
      MPI_Allreduce(&maxdepth, &maxdepth_glb, 1, MPI_INT, MPI_MAX, comm);
    }
    if(!myrank) std::cout<<"Number of Leaf Nodes: "<<nleaf_glb<<'\n';
    if(!myrank) std::cout<<"Tree Depth: "<<maxdepth_glb<<'\n';
  }

  // Setup FMM
  tree->SetupFMM(fmm_mat);
  tree->RunFMM();

  //Re-run FMM
  tree->ClearFMMData();
  tree->RunFMM();

  //Find error in FMM output.
  CheckFMMOutput<FMM_Mat_t>(tree, mykernel, "Output");

  //Write2File
  //tree->Write2File("result/output");

  //Delete matrices.
  delete fmm_mat;

  //Delete the tree.
  delete tree;
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);

  MPI_Comm comm=MPI_COMM_WORLD;

  // Read command line options.
  commandline_option_start(argc, argv);
  omp_set_num_threads( atoi(commandline_option(argc, argv,  "-omp",     "1", false, "-omp  <int> =  (1)   : Number of OpenMP threads."          )));
  size_t   N=(size_t)strtod(commandline_option(argc, argv,    "-N",     "1",  true, "-N    <int>          : Number of points."                  ),NULL);
  size_t   M=(size_t)strtod(commandline_option(argc, argv,    "-M",   "350", false, "-M    <int>          : Number of points per octant."       ),NULL);
  double   b=        strtod(commandline_option(argc, argv,    "-b",     "1", false, "-b    <int> =  (1)   : Bounding-box length (0 < b <= 1)"   ),NULL);
  int      m=       strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  int      d=       strtoul(commandline_option(argc, argv,    "-d",    "15", false, "-d    <int> = (15)   : Maximum tree depth."                ),NULL,10);
  bool    sp=   (1==strtoul(commandline_option(argc, argv,   "-sp",     "0", false, "-sp   <0/1> =  (0)   : Single Precision."                  ),NULL,10));
  int   dist=       strtoul(commandline_option(argc, argv, "-dist",     "0", false, "-dist <int> =  (0)   : 0) Unif 1) Sphere 2) Ellipse"       ),NULL,10);
  int    ker=       strtoul(commandline_option(argc, argv,  "-ker",     "1", false,
       "-ker  <int> =  (1)   : 1) Laplace potential\n\
                               2) Laplace gradient\n\
                               3) Helmholtz"),NULL,10);
  commandline_option_end(argc, argv);
  pvfmm::Profile::Enable(true);

  // Run FMM with above options.
  pvfmm::Profile::Tic("FMM_Test",&comm,true);
  if(sp) fmm_test<float >(ker, N,M,b, dist,m,d,comm);
  else   fmm_test<double>(ker, N,M,b, dist,m,d,comm);
  pvfmm::Profile::Toc();

  //Output Profiling results.
  pvfmm::Profile::print(&comm);

  // Shut down MPI
  MPI_Finalize();
  return 0;
}

