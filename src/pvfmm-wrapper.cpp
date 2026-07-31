#include <cstring>
#include <pvfmm.hpp>
#include <pvfmm.h>

#if !defined(SCTL_HAVE_MPI)
// MPI unavailable: no-op stand-ins so the C/Fortran wrapper still builds.
// Communicator handles are ignored (single-process / self-communicator).
#define MPI_COMM_WORLD ((MPI_Comm)0)
static inline MPI_Comm MPI_Comm_f2c(MPI_Fint f) { (void)f; return (MPI_Comm)0; }
#endif

#if defined(SCTL_HAVE_MPI)
// pvfmm.f90 and the dlopen bindings pass Fortran communicator handles as
// plain 32-bit ints (integer(c_int)); MPI_Comm_f2c must agree.
static_assert(sizeof(MPI_Fint) == sizeof(int), "MPI_Fint is expected to be int-sized");
#endif

// Build an sctl::Comm from a C-API communicator handle (self when built w/o MPI).
static inline sctl::Comm PVFMMComm(MPI_Comm comm) {
#if defined(SCTL_HAVE_MPI)
  return sctl::Comm(comm);
#else
  (void)comm;
  return sctl::Comm();
#endif
}

#ifdef __cplusplus
extern "C" { // Volume FM
#endif

// Map the C-API boundary type to pvfmm::BoundaryType. Takes int so the
// Fortran shims can pass the int32 flag through; values 0/1 also cover the
// boolean periodic flag accepted by earlier versions of the C API.
static pvfmm::BoundaryType PVFMMBndry(int bndry) {
  switch (bndry) {
    case PVFMMBoundaryPXYZ: return pvfmm::PXYZ;
    case PVFMMBoundaryPX:   return pvfmm::PX;
    case PVFMMBoundaryPXY:  return pvfmm::PXY;
    default:                return pvfmm::FreeSpace;
  }
}

static void PVFMMEnsureMPIInitialized() {
#if defined(SCTL_HAVE_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
  }
#endif
}

void* PVFMMCreateVolumeFMMF(int m, int q, enum PVFMMKernel kernel, MPI_Comm comm) {
  const pvfmm::Kernel<float>* ker = nullptr;
  if (kernel == PVFMMLaplacePotential   ) ker = &pvfmm::LaplaceKernel   <float>::potential();
  if (kernel == PVFMMLaplaceGradient    ) ker = &pvfmm::LaplaceKernel   <float>::gradient();
  if (kernel == PVFMMStokesPressure     ) ker = &pvfmm::StokesKernel    <float>::pressure();
  if (kernel == PVFMMStokesVelocity     ) ker = &pvfmm::StokesKernel    <float>::velocity();
  if (kernel == PVFMMStokesVelocityGrad ) ker = &pvfmm::StokesKernel    <float>::vel_grad();
  if (kernel == PVFMMBiotSavartPotential) ker = &pvfmm::BiotSavartKernel<float>::potential();

  pvfmm::ChebFMM<float>* matrices = new pvfmm::ChebFMM<float>;
  matrices->Initialize(m, q, PVFMMComm(comm), ker);
  return (void*)matrices;
}

void* PVFMMCreateVolumeTreeF(int cheb_deg, int data_dim, void (*fn_ptr)(const float* coord, long n, float* out, const void* ctx), const void* fn_ctx, const float* trg_coord, long n_trg, MPI_Comm comm, float tol, int max_pts, enum PVFMMBoundaryType bndry, int init_depth) {
  const int COORD_DIM = 3;
  std::vector<float> trg_coord_(n_trg*COORD_DIM);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n_trg*COORD_DIM; i++) trg_coord_[i] = trg_coord[i];

  std::function<void(const float*,int,float*)> fn_ptr_ = [&fn_ctx,&fn_ptr](const float* coord, int n, float* out) {
    fn_ptr(coord, n, out, fn_ctx);
  };

  auto* tree = ChebFMM_CreateTree(cheb_deg, data_dim, fn_ptr_, trg_coord_, PVFMMComm(comm), tol, max_pts, PVFMMBndry(bndry), init_depth);
  //tree->Write2File("vis",4);

  return (void*)tree;
}

void* PVFMMCreateVolumeTreeFromCoeffF(long n_nodes, int cheb_deg, int data_dim, const float* node_coord, const float* fn_coeff, const float* trg_coord, long n_trg, MPI_Comm comm, enum PVFMMBoundaryType bndry) {
  const int COORD_DIM = 3;
  std::vector<float> node_coord_(n_nodes*COORD_DIM), fn_coeff_(n_nodes*data_dim*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6), trg_coord_(n_trg*COORD_DIM);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_coord_.size(); i++) node_coord_[i] = node_coord[i];
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)fn_coeff_.size(); i++) fn_coeff_[i] = fn_coeff[i];
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)trg_coord_.size(); i++) trg_coord_[i] = trg_coord[i];

  auto* tree = ChebFMM_CreateTree(cheb_deg, node_coord_, fn_coeff_, trg_coord_, PVFMMComm(comm), PVFMMBndry(bndry));
  //tree->Write2File("vis",4);

  return (void*)tree;
}

void PVFMMEvaluateVolumeFMMF(float* trg_val, void* tree, const void* fmm, long loc_size) {
  auto* tree_ = (pvfmm::ChebFMM_Tree<float>*)tree;
  tree_->SetupFMM((pvfmm::ChebFMM<float>*)fmm);

  std::vector<float> trg_val_;
  ChebFMM_Evaluate(trg_val_, tree_, loc_size);
  for (long i = 0; i < (long)trg_val_.size(); i++) trg_val[i] = trg_val_[i];
}

void PVFMMDestroyVolumeFMMF(void** ctx) {
  if(!ctx[0]) return;
  delete (pvfmm::ChebFMM<float>*)ctx[0];
  ctx[0]=NULL;
}

void PVFMMDestroyVolumeTreeF(void** ctx) {
  if(!ctx[0]) return;
  delete (pvfmm::ChebFMM_Tree<float>*)ctx[0];
  ctx[0]=NULL;
}

long PVFMMGetLeafCountF(const void* tree) {
  std::vector<float> node_coord_;
  ChebFMM_GetLeafCoord(node_coord_, (pvfmm::ChebFMM_Tree<float>*)tree);
  return (long)node_coord_.size()/3;
}

void PVFMMGetLeafCoordF(float* node_coord, const void* tree) {
  std::vector<float> node_coord_;
  ChebFMM_GetLeafCoord(node_coord_, (pvfmm::ChebFMM_Tree<float>*)tree);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_coord_.size(); i++) node_coord[i] = node_coord_[i];
}

void PVFMMGetPotentialCoeffF(float* coeff, const void* tree) {
  std::vector<float> coeff_;
  ChebFMM_GetPotentialCoeff(coeff_, (pvfmm::ChebFMM_Tree<float>*)tree);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)coeff_.size(); i++) coeff[i] = coeff_[i];
}

void PVFMMCoeff2NodesF(float* node_val, long Nleaf, int ChebDeg, int dof, const float* coeff) {
  std::vector<float> node_val_, coeff_(Nleaf*dof*(ChebDeg+1)*(ChebDeg+2)*(ChebDeg+3)/6);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)coeff_.size(); i++) coeff_[i] = coeff[i];
  pvfmm::ChebFMM_Coeff2Nodes(node_val_, ChebDeg, dof, coeff_);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_val_.size(); i++) node_val[i] = node_val_[i];
}

void PVFMMNodes2CoeffF(float* coeff, long Nleaf, int ChebDeg, int dof, const float* node_val) {
  std::vector<float> node_val_(Nleaf*(ChebDeg+1)*(ChebDeg+1)*(ChebDeg+1)*dof), coeff_;
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_val_.size(); i++) node_val_[i] = node_val[i];
  pvfmm::ChebFMM_Nodes2Coeff(coeff_, ChebDeg, dof, node_val_);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)coeff_.size(); i++) coeff[i] = coeff_[i];
}




void* PVFMMCreateVolumeFMMD(int m, int q, enum PVFMMKernel kernel, MPI_Comm comm) {
  const pvfmm::Kernel<double>* ker = nullptr;
  if (kernel == PVFMMLaplacePotential   ) ker = &pvfmm::LaplaceKernel   <double>::potential();
  if (kernel == PVFMMLaplaceGradient    ) ker = &pvfmm::LaplaceKernel   <double>::gradient();
  if (kernel == PVFMMStokesPressure     ) ker = &pvfmm::StokesKernel    <double>::pressure();
  if (kernel == PVFMMStokesVelocity     ) ker = &pvfmm::StokesKernel    <double>::velocity();
  if (kernel == PVFMMStokesVelocityGrad ) ker = &pvfmm::StokesKernel    <double>::vel_grad();
  if (kernel == PVFMMBiotSavartPotential) ker = &pvfmm::BiotSavartKernel<double>::potential();

  pvfmm::ChebFMM<double>* matrices = new pvfmm::ChebFMM<double>;
  matrices->Initialize(m, q, PVFMMComm(comm), ker);
  return (void*)matrices;
}

void* PVFMMCreateVolumeTreeD(int cheb_deg, int data_dim, void (*fn_ptr)(const double* coord, long n, double* out, const void* ctx), const void* fn_ctx, const double* trg_coord, long n_trg, MPI_Comm comm, double tol, int max_pts, enum PVFMMBoundaryType bndry, int init_depth) {
  const int COORD_DIM = 3;
  std::vector<double> trg_coord_(n_trg*COORD_DIM);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n_trg*COORD_DIM; i++) trg_coord_[i] = trg_coord[i];

  std::function<void(const double*,int,double*)> fn_ptr_ = [&fn_ctx,&fn_ptr](const double* coord, int n, double* out) {
    fn_ptr(coord, n, out, fn_ctx);
  };

  auto* tree = ChebFMM_CreateTree(cheb_deg, data_dim, fn_ptr_, trg_coord_, PVFMMComm(comm), tol, max_pts, PVFMMBndry(bndry), init_depth);
  //tree->Write2File("vis",4);

  return (void*)tree;
}

void* PVFMMCreateVolumeTreeFromCoeffD(long n_nodes, int cheb_deg, int data_dim, const double* node_coord, const double* fn_coeff, const double* trg_coord, long n_trg, MPI_Comm comm, enum PVFMMBoundaryType bndry) {
  const int COORD_DIM = 3;
  std::vector<double> node_coord_(n_nodes*COORD_DIM), fn_coeff_(n_nodes*data_dim*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6), trg_coord_(n_trg*COORD_DIM);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_coord_.size(); i++) node_coord_[i] = node_coord[i];
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)fn_coeff_.size(); i++) fn_coeff_[i] = fn_coeff[i];
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)trg_coord_.size(); i++) trg_coord_[i] = trg_coord[i];

  auto* tree = ChebFMM_CreateTree(cheb_deg, node_coord_, fn_coeff_, trg_coord_, PVFMMComm(comm), PVFMMBndry(bndry));
  //tree->Write2File("vis",4);

  return (void*)tree;
}

void PVFMMEvaluateVolumeFMMD(double* trg_val, void* tree, const void* fmm, long loc_size) {
  auto* tree_ = (pvfmm::ChebFMM_Tree<double>*)tree;
  tree_->SetupFMM((pvfmm::ChebFMM<double>*)fmm);

  std::vector<double> trg_val_;
  ChebFMM_Evaluate(trg_val_, tree_, loc_size);
  for (long i = 0; i < (long)trg_val_.size(); i++) trg_val[i] = trg_val_[i];
}

void PVFMMDestroyVolumeFMMD(void** ctx) {
  if(!ctx[0]) return;
  delete (pvfmm::ChebFMM<double>*)ctx[0];
  ctx[0]=NULL;
}

void PVFMMDestroyVolumeTreeD(void** ctx) {
  if(!ctx[0]) return;
  delete (pvfmm::ChebFMM_Tree<double>*)ctx[0];
  ctx[0]=NULL;
}

long PVFMMGetLeafCountD(const void* tree) {
  std::vector<double> node_coord_;
  ChebFMM_GetLeafCoord(node_coord_, (pvfmm::ChebFMM_Tree<double>*)tree);
  return (long)node_coord_.size()/3;
}

void PVFMMGetLeafCoordD(double* node_coord, const void* tree) {
  std::vector<double> node_coord_;
  ChebFMM_GetLeafCoord(node_coord_, (pvfmm::ChebFMM_Tree<double>*)tree);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_coord_.size(); i++) node_coord[i] = node_coord_[i];
}

void PVFMMGetPotentialCoeffD(double* coeff, const void* tree) {
  std::vector<double> coeff_;
  ChebFMM_GetPotentialCoeff(coeff_, (pvfmm::ChebFMM_Tree<double>*)tree);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)coeff_.size(); i++) coeff[i] = coeff_[i];
}

void PVFMMCoeff2NodesD(double* node_val, long Nleaf, int ChebDeg, int dof, const double* coeff) {
  std::vector<double> node_val_, coeff_(Nleaf*dof*(ChebDeg+1)*(ChebDeg+2)*(ChebDeg+3)/6);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)coeff_.size(); i++) coeff_[i] = coeff[i];
  pvfmm::ChebFMM_Coeff2Nodes(node_val_, ChebDeg, dof, coeff_);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_val_.size(); i++) node_val[i] = node_val_[i];
}

void PVFMMNodes2CoeffD(double* coeff, long Nleaf, int ChebDeg, int dof, const double* node_val) {
  std::vector<double> node_val_(Nleaf*(ChebDeg+1)*(ChebDeg+1)*(ChebDeg+1)*dof), coeff_;
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)node_val_.size(); i++) node_val_[i] = node_val[i];
  pvfmm::ChebFMM_Nodes2Coeff(coeff_, ChebDeg, dof, node_val_);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)coeff_.size(); i++) coeff[i] = coeff_[i];
}




void pvfmmcreatevolumefmmf_(void** ctx, const int32_t* m, const int32_t* q, const int32_t* kernel, const MPI_Fint* fcomm) {
  const MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  (*ctx) = PVFMMCreateVolumeFMMF(*m, *q, (PVFMMKernel)*kernel, comm);
}

void pvfmmdestroyvolumefmmf_(void** ctx) {
  PVFMMDestroyVolumeFMMF(ctx);
}

void pvfmmcreatevolumetreef_(void** ctx, const int32_t* cheb_deg, const int32_t* data_dim, void (*fn_ptr)(const float* coord, const int64_t* n, float* out), const float* trg_coord, const int64_t* n_trg, const MPI_Fint* fcomm, const float* tol, const int32_t* max_pts, const int32_t* periodic, const int32_t* init_depth) {
  const int COORD_DIM = 3;
  std::vector<float> trg_coord_((*n_trg)*COORD_DIM);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (*n_trg)*COORD_DIM; i++) trg_coord_[i] = trg_coord[i];

  std::function<void(const float*,int,float*)> fn_ptr_ = [&fn_ptr](const float* coord, int64_t n, float* out) {
    fn_ptr(coord, &n, out);
  };

  const MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  auto* tree = ChebFMM_CreateTree(*cheb_deg, *data_dim, fn_ptr_, trg_coord_, PVFMMComm(comm), *tol, *max_pts, PVFMMBndry(*periodic), *init_depth);
  //tree->Write2File("vis",4);

  (*ctx) = (void*)tree;
}

void pvfmmcreatevolumetreefromcoefff_(void** ctx, const int64_t* n_nodes, const int32_t* cheb_deg, const int32_t* data_dim, const float* node_coord, const float* fn_coeff, const float* trg_coord, const int64_t* n_trg, const MPI_Fint* fcomm, const int32_t* periodic) {
  const MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  (*ctx) = PVFMMCreateVolumeTreeFromCoeffF(*n_nodes, *cheb_deg, *data_dim, node_coord, fn_coeff, trg_coord, *n_trg, comm, (enum PVFMMBoundaryType)*periodic);
}

void pvfmmdestroyvolumetreef_(void** ctx) {
  PVFMMDestroyVolumeTreeF(ctx);
}

void pvfmmevaluatevolumefmmf_(float* trg_val, void** tree, const void** fmm, const int64_t* loc_size) {
  PVFMMEvaluateVolumeFMMF(trg_val, *tree, *fmm, *loc_size);
}

void pvfmmgetleafcountf_(int64_t* Nleaf, const void** tree) {
  (*Nleaf) = PVFMMGetLeafCountF(*tree);
}

void pvfmmgetleafcoordf_(float* node_coord, const void** tree) {
  PVFMMGetLeafCoordF(node_coord, *tree);
}

void pvfmmgetpotentialcoefff_(float* coeff, const void** tree) {
  PVFMMGetPotentialCoeffF(coeff, *tree);
}

void pvfmmcoeff2nodesf_(float* node_val, const int64_t* Nleaf, const int32_t* ChebDeg, const int32_t* dof, const float* coeff) {
  PVFMMCoeff2NodesF(node_val, *Nleaf, *ChebDeg, *dof, coeff);
}

void pvfmmnodes2coefff_(float* coeff, const int64_t* Nleaf, const int32_t* ChebDeg, const int32_t* dof, const float* node_val) {
  PVFMMNodes2CoeffF(coeff, *Nleaf, *ChebDeg, *dof, node_val);
}




void pvfmmcreatevolumefmmd_(void** ctx, const int32_t* m, const int32_t* q, const int32_t* kernel, const MPI_Fint* fcomm) {
  const MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  (*ctx) = PVFMMCreateVolumeFMMD(*m, *q, (PVFMMKernel)*kernel, comm);
}

void pvfmmdestroyvolumefmmd_(void** ctx) {
  PVFMMDestroyVolumeFMMD(ctx);
}

void pvfmmcreatevolumetreed_(void** ctx, const int32_t* cheb_deg, const int32_t* data_dim, void (*fn_ptr)(const double* coord, const int64_t* n, double* out), const double* trg_coord, const int64_t* n_trg, const MPI_Fint* fcomm, const double* tol, const int32_t* max_pts, const int32_t* periodic, const int32_t* init_depth) {
  const int COORD_DIM = 3;
  std::vector<double> trg_coord_((*n_trg)*COORD_DIM);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (*n_trg)*COORD_DIM; i++) trg_coord_[i] = trg_coord[i];

  std::function<void(const double*,int,double*)> fn_ptr_ = [&fn_ptr](const double* coord, int64_t n, double* out) {
    fn_ptr(coord, &n, out);
  };

  const MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  auto* tree = ChebFMM_CreateTree(*cheb_deg, *data_dim, fn_ptr_, trg_coord_, PVFMMComm(comm), *tol, *max_pts, PVFMMBndry(*periodic), *init_depth);
  //tree->Write2File("vis",4);

  (*ctx) = (void*)tree;
}

void pvfmmcreatevolumetreefromcoeffd_(void** ctx, const int64_t* n_nodes, const int32_t* cheb_deg, const int32_t* data_dim, const double* node_coord, const double* fn_coeff, const double* trg_coord, const int64_t* n_trg, const MPI_Fint* fcomm, const int32_t* periodic) {
  const MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  (*ctx) = PVFMMCreateVolumeTreeFromCoeffD(*n_nodes, *cheb_deg, *data_dim, node_coord, fn_coeff, trg_coord, *n_trg, comm, (enum PVFMMBoundaryType)*periodic);
}

void pvfmmdestroyvolumetreed_(void** ctx) {
  PVFMMDestroyVolumeTreeD(ctx);
}

void pvfmmevaluatevolumefmmd_(double* trg_val, void** tree, const void** fmm, const int64_t* loc_size) {
  PVFMMEvaluateVolumeFMMD(trg_val, *tree, *fmm, *loc_size);
}

void pvfmmgetleafcountd_(int64_t* Nleaf, const void** tree) {
  (*Nleaf) = PVFMMGetLeafCountD(*tree);
}

void pvfmmgetleafcoordd_(double* node_coord, const void** tree) {
  PVFMMGetLeafCoordD(node_coord, *tree);
}

void pvfmmgetpotentialcoeffd_(double* coeff, const void** tree) {
  PVFMMGetPotentialCoeffD(coeff, *tree);
}

void pvfmmcoeff2nodesd_(double* node_val, const int64_t* Nleaf, const int32_t* ChebDeg, const int32_t* dof, const double* coeff) {
  PVFMMCoeff2NodesD(node_val, *Nleaf, *ChebDeg, *dof, coeff);
}

void pvfmmnodes2coeffd_(double* coeff, const int64_t* Nleaf, const int32_t* ChebDeg, const int32_t* dof, const double* node_val) {
  PVFMMNodes2CoeffD(coeff, *Nleaf, *ChebDeg, *dof, node_val);
}

#ifdef __cplusplus
}
#endif



template<typename Real> struct PVFMMContext{
  typedef pvfmm::FMM_Node<pvfmm::MPI_Node<Real> > Node_t;
  typedef pvfmm::FMM_Pts<Node_t> Mat_t;
  typedef pvfmm::FMM_Tree<Mat_t> Tree_t;

  Real box_size;
  int max_pts;
  int mult_order;
  int max_depth;
  pvfmm::BoundaryType bndry;
  const pvfmm::Kernel<Real>* ker;
  MPI_Comm comm;            // raw handle kept for the C/Fortran API boundary.
  sctl::Comm sctl_comm;     // sctl wrapper around `comm`, used by sctl::Profile.

  typename Node_t::NodeData tree_data;
  Tree_t* tree;
  Mat_t* mat;
};

// Kernel lookup for the C-API enum.
template <typename Real> static const pvfmm::Kernel<Real>* PVFMMKernelPtr(enum PVFMMKernel kernel) {
  if (kernel == PVFMMLaplacePotential   ) return &pvfmm::LaplaceKernel   <Real>::potential();
  if (kernel == PVFMMLaplaceGradient    ) return &pvfmm::LaplaceKernel   <Real>::gradient();
  if (kernel == PVFMMStokesPressure     ) return &pvfmm::StokesKernel    <Real>::pressure();
  if (kernel == PVFMMStokesVelocity     ) return &pvfmm::StokesKernel    <Real>::velocity();
  if (kernel == PVFMMStokesVelocityGrad ) return &pvfmm::StokesKernel    <Real>::vel_grad();
  if (kernel == PVFMMBiotSavartPotential) return &pvfmm::BiotSavartKernel<Real>::potential();
  return nullptr;
}

template<typename Real> static void* PVFMMCreateContext(Real box_size, int n, int m, int max_d, const pvfmm::Kernel<Real>* ker, pvfmm::BoundaryType bndry, MPI_Comm comm) {
  if (bndry != pvfmm::BoundaryType::FreeSpace && box_size <= 0) {
    fprintf(stderr, "PVFMMCreateContext error: periodic boundary conditions require box_size > 0\n");
    return nullptr;
  }
  PVFMMEnsureMPIInitialized();
  const sctl::Comm sctl_comm = PVFMMComm(comm);
  sctl::Profile::Tic("FMMContext", &sctl_comm, true);
  bool prof_state=sctl::Profile::Enable(false);

  // Create new context.
  PVFMMContext<Real>* ctx = new PVFMMContext<Real>;

  // Set member variables.
  ctx->box_size=box_size;
  ctx->max_pts=n;
  ctx->mult_order=m;
  ctx->max_depth=max_d;
  ctx->bndry=bndry;
  ctx->ker=ker;
  ctx->comm=comm;
  ctx->sctl_comm=sctl_comm;  // share ref-counted Impl, no extra MPI_Comm_dup.

  // Initialize FMM matrices.
  ctx->mat=new typename PVFMMContext<Real>::Mat_t();
  ctx->mat->Initialize(ctx->mult_order, PVFMMComm(ctx->comm), ctx->ker);

  // Set tree_data
  ctx->tree_data.dim=PVFMM_COORD_DIM;
  ctx->tree_data.max_depth=ctx->max_depth;
  ctx->tree_data.max_pts=ctx->max_pts;
  { // ctx->tree_data.pt_coord=... //Set points for initial tree.
    int np, myrank;
    np = ctx->sctl_comm.Size();
    myrank = ctx->sctl_comm.Rank();

    std::vector<Real> coord;
    size_t NN=(size_t)ceil(pow((Real)np*ctx->max_pts,1.0/3.0));
    size_t N_total=NN*NN*NN;
    size_t start= myrank   *N_total/np;
    size_t end  =(myrank+1)*N_total/np;
    for(size_t i=start;i<end;i++){
      coord.push_back((Real)(((i/  1    )%NN)+0.5)/NN);
      coord.push_back((Real)(((i/ NN    )%NN)+0.5)/NN);
      coord.push_back((Real)(((i/(NN*NN))%NN)+0.5)/NN);
    }
    ctx->tree_data.pt_coord=coord;
  }

  // Construct tree.
  bool adap=false; // no data to do adaptive.
  ctx->tree=new typename PVFMMContext<Real>::Tree_t(PVFMMComm(comm));
  ctx->tree->Initialize(&ctx->tree_data);
  ctx->tree->InitFMM_Tree(adap,ctx->bndry);

  sctl::Profile::Enable(prof_state);
  sctl::Profile::Toc();
  return ctx;
}

template<typename Real> static void PVFMMEval(const Real* src_pos, const Real* sl_den, const Real* dl_den, size_t n_src, const Real* trg_pos, Real* trg_val, size_t n_trg, const void* ctx_, int setup){
  size_t omp_p=omp_get_max_threads();

  typedef pvfmm::FMM_Node<pvfmm::MPI_Node<Real> > Node_t;
  //typedef pvfmm::FMM_Pts<Node_t> Mat_t;
  //typedef pvfmm::FMM_Tree<Mat_t> Tree_t;

  assert(ctx_);
  PVFMMContext<Real>* ctx=(PVFMMContext<Real>*)ctx_;
  const int* ker_dim=ctx->ker->ker_dim;

  sctl::Profile::Tic("FMM", &ctx->sctl_comm);
  Real scale_x, shift_x[PVFMM_COORD_DIM];
  if(ctx->box_size<=0){ // determine bounding box
    Real s0, x0[PVFMM_COORD_DIM];
    Real s1, x1[PVFMM_COORD_DIM];

    auto PVFMMBoundingBox = [](size_t n_src, const Real* x, Real* scale_xr, Real* shift_xr, const sctl::Comm& comm){
      Real& scale_x=*scale_xr;
      Real* shift_x= shift_xr;

      assert(n_src>0);
      { // Compute bounding box
        sctl::StaticArray<double,PVFMM_COORD_DIM> loc_min_x;
        sctl::StaticArray<double,PVFMM_COORD_DIM> loc_max_x;
        assert(n_src>0);
        for(size_t k=0;k<PVFMM_COORD_DIM;k++){
          loc_min_x[k]=loc_max_x[k]=x[k];
        }

        for(size_t i=0;i<n_src;i++){
          const Real* x_=&x[i*PVFMM_COORD_DIM];
          for(size_t k=0;k<PVFMM_COORD_DIM;k++){
            if(loc_min_x[k]>x_[0]) loc_min_x[k]=x_[0];
            if(loc_max_x[k]<x_[0]) loc_max_x[k]=x_[0];
            ++x_;
          }
        }

        sctl::StaticArray<double,PVFMM_COORD_DIM> min_x;
        sctl::StaticArray<double,PVFMM_COORD_DIM> max_x;
        comm.Allreduce<double>(loc_min_x, min_x, PVFMM_COORD_DIM, sctl::CommOp::MIN);
        comm.Allreduce<double>(loc_max_x, max_x, PVFMM_COORD_DIM, sctl::CommOp::MAX);

        Real eps=sctl::machine_eps<Real>()*64; // Points should be well within the box.
        scale_x=1/(Real)(max_x[0]-min_x[0]+2*eps);
        for(size_t k=0;k<PVFMM_COORD_DIM;k++){
          scale_x=std::min(scale_x,(Real)(1.0/(max_x[k]-min_x[k]+2*eps)));
        }
        if(scale_x*0.0!=0.0) scale_x=1.0; // fix for scal_x=inf
        for(size_t k=0;k<PVFMM_COORD_DIM;k++){
          shift_x[k]=(Real)-min_x[k]*scale_x+eps;
        }
      }
    };
    PVFMMBoundingBox(n_src, src_pos, &s0, x0, ctx->sctl_comm);
    PVFMMBoundingBox(n_trg, trg_pos, &s1, x1, ctx->sctl_comm);

    Real c0[PVFMM_COORD_DIM]={(Real)(0.5-x0[0])/s0, (Real)(0.5-x0[1])/s0, (Real)(0.5-x0[2])/s0};
    Real c1[PVFMM_COORD_DIM]={(Real)(0.5-x1[0])/s1, (Real)(0.5-x1[1])/s1, (Real)(0.5-x1[2])/s1};

    scale_x=0;
    scale_x=std::max<Real>(scale_x, sctl::fabs<Real>(c0[0]-c1[0]));
    scale_x=std::max<Real>(scale_x, sctl::fabs<Real>(c0[1]-c1[1]));
    scale_x=std::max<Real>(scale_x, sctl::fabs<Real>(c0[2]-c1[2]));
    scale_x=1/(scale_x+1/s0+1/s1);

    shift_x[0]=(Real)0.5-(c0[0]+c1[0])*scale_x/2;
    shift_x[1]=(Real)0.5-(c0[1]+c1[1])*scale_x/2;
    shift_x[2]=(Real)0.5-(c0[2]+c1[2])*scale_x/2;
  }else{
    scale_x=1/ctx->box_size;
    shift_x[0]=0;
    shift_x[1]=0;
    shift_x[2]=0;
  }

  sctl::Vector<Real>  src_scal;
  sctl::Vector<Real>  trg_scal;
  sctl::Vector<Real> surf_scal;
  { // Set src_scal, trg_scal
    sctl::Vector<Real>& src_scal_exp=ctx->ker->src_scal;
    sctl::Vector<Real>& trg_scal_exp=ctx->ker->trg_scal;
    src_scal .ReInit(ctx->ker->src_scal.Dim());
    trg_scal .ReInit(ctx->ker->trg_scal.Dim());
    surf_scal.ReInit(PVFMM_COORD_DIM+src_scal.Dim());
    for(sctl::Long i=0;i<src_scal.Dim();i++){
      src_scal [i]=sctl::pow(scale_x, src_scal_exp[i]);
      surf_scal[i]=scale_x*src_scal[i];
    }
    for(sctl::Long i=0;i<trg_scal.Dim();i++){
      trg_scal[i]=sctl::pow(scale_x, trg_scal_exp[i]);
    }
    for(sctl::Long i=src_scal.Dim();i<surf_scal.Dim();i++){
      surf_scal[i]=1;
    }
  }

  sctl::Vector<sctl::Long> scatter_index;
  { // Set tree_data
    sctl::Vector<Real>&  trg_coord=ctx->tree_data. trg_coord;
    sctl::Vector<Real>&  src_coord=ctx->tree_data. src_coord;
    sctl::Vector<Real>&  src_value=ctx->tree_data. src_value;
    sctl::Vector<Real>& surf_value=ctx->tree_data.surf_value;
    sctl::Vector<pvfmm::MortonId> pt_mid;

    std::vector<Node_t*> nodes;
    { // Get list of leaf nodes.
      std::vector<sctl::Iterator<Node_t>>& all_nodes=ctx->tree->GetNodeList();
      for(size_t i=0;i<all_nodes.size();i++){
        if(all_nodes[i]->IsLeaf() && !all_nodes[i]->IsGhost()){
          nodes.push_back(&all_nodes[i][0]);
        }
      }
    }

    pvfmm::MortonId min_mid;
    { // Get first MortonId
      sctl::Iterator<Node_t> n=ctx->tree->PreorderFirst();
      while(n!=sctl::NullIterator<Node_t>()){
        if(!n->IsGhost() && n->IsLeaf()) break;
        n=ctx->tree->PreorderNxt(n);
      }
      assert(n!=sctl::NullIterator<Node_t>());
      min_mid=n->GetMortonId();
    }

    { // Set src tree_data
      { // Scatter src data
        // Compute MortonId and copy coordinates and values.
        src_coord .ReInit(       n_src            *PVFMM_COORD_DIM);
        src_value .ReInit(sl_den?n_src*(ker_dim[0]          ):0);
        surf_value.ReInit(dl_den?n_src*(ker_dim[0]+PVFMM_COORD_DIM):0);
        pt_mid    .ReInit(n_src);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          size_t a=((tid+0)*n_src)/omp_p;
          size_t b=((tid+1)*n_src)/omp_p;
          for(size_t i=a;i<b;i++){
            for(size_t j=0;j<PVFMM_COORD_DIM;j++){
              src_coord[i*PVFMM_COORD_DIM+j]=src_pos[i*PVFMM_COORD_DIM+j]*scale_x+shift_x[j];
              while(src_coord[i*PVFMM_COORD_DIM+j]< 0.0) src_coord[i*PVFMM_COORD_DIM+j]+=1;
              while(src_coord[i*PVFMM_COORD_DIM+j]>=1.0) src_coord[i*PVFMM_COORD_DIM+j]-=1;
            }
            pt_mid[i]=pvfmm::MortonId(sctl::Ptr2ConstItr<Real>(&src_coord[i*PVFMM_COORD_DIM],3));
          }
          if(src_value.Dim()) for(size_t i=a;i<b;i++){
            for(int j=0;j<ker_dim[0];j++){
              src_value[i*ker_dim[0]+j]=sl_den[i*ker_dim[0]+j]*src_scal[j];
            }
          }
          if(surf_value.Dim()) for(size_t i=a;i<b;i++){
            for(int j=0;j<ker_dim[0]+PVFMM_COORD_DIM;j++){
              surf_value[i*(ker_dim[0]+PVFMM_COORD_DIM)+j]=dl_den[i*(ker_dim[0]+PVFMM_COORD_DIM)+j]*surf_scal[j];
            }
          }
        }

        // Scatter src coordinates and values.
        ctx->sctl_comm.SortScatterIndex( pt_mid  , scatter_index, &min_mid);
        ctx->sctl_comm.ScatterForward( pt_mid  , scatter_index);
        ctx->sctl_comm.ScatterForward(src_coord, scatter_index);
        if( src_value.Dim()) ctx->sctl_comm.ScatterForward( src_value, scatter_index);
        if(surf_value.Dim()) ctx->sctl_comm.ScatterForward(surf_value, scatter_index);
      }
      { // Set src tree_data
        std::vector<size_t> part_indx(nodes.size()+1);
        part_indx[nodes.size()]=pt_mid.Dim();
        #pragma omp parallel for
        for(size_t j=0;j<nodes.size();j++){
          part_indx[j]=std::lower_bound(pt_mid.begin(), pt_mid.begin()+pt_mid.Dim(), nodes[j]->GetMortonId())-pt_mid.begin();
        }

        if(setup){
          #pragma omp parallel for
          for(size_t j=0;j<nodes.size();j++){
            size_t n_pts=part_indx[j+1]-part_indx[j];
            if(src_value.Dim()){
              nodes[j]-> src_coord.ReInit(n_pts*( PVFMM_COORD_DIM), src_coord.begin()+part_indx[j]*( PVFMM_COORD_DIM),false);
              nodes[j]-> src_value.ReInit(n_pts*(ker_dim[0]), src_value.begin()+part_indx[j]*(ker_dim[0]),false);
            }else{
              nodes[j]-> src_coord.ReInit(0,sctl::NullIterator<Real>(),false);
              nodes[j]-> src_value.ReInit(0,sctl::NullIterator<Real>(),false);
            }
            if(surf_value.Dim()){
              nodes[j]->surf_coord.ReInit(n_pts*(           PVFMM_COORD_DIM), src_coord.begin()+part_indx[j]*(           PVFMM_COORD_DIM),false);
              nodes[j]->surf_value.ReInit(n_pts*(ker_dim[0]+PVFMM_COORD_DIM), surf_value.begin()+part_indx[j]*(ker_dim[0]+PVFMM_COORD_DIM),false);
            }else{
              nodes[j]->surf_coord.ReInit(0,sctl::NullIterator<Real>(),false);
              nodes[j]->surf_value.ReInit(0,sctl::NullIterator<Real>(),false);
            }
          }
        }else{
          #pragma omp parallel for
          for(size_t j=0;j<nodes.size();j++){
            size_t n_pts=part_indx[j+1]-part_indx[j];
            if(src_value.Dim()){
              assert((size_t)nodes[j]->src_coord.Dim()==n_pts*( PVFMM_COORD_DIM));
              assert((size_t)nodes[j]->src_value.Dim()==n_pts*(ker_dim[0]));
              //memcpy(&nodes[j]->src_coord[0],&src_coord[0]+part_indx[j]*( PVFMM_COORD_DIM),n_pts*( PVFMM_COORD_DIM)*sizeof(Real));
              memcpy(nodes[j]->src_value.begin(),src_value.begin()+part_indx[j]*(ker_dim[0]),n_pts*(ker_dim[0])*sizeof(Real));
            }
            if(surf_value.Dim()){
              assert((size_t)nodes[j]->surf_coord.Dim()==n_pts*(           PVFMM_COORD_DIM));
              assert((size_t)nodes[j]->surf_value.Dim()==n_pts*(ker_dim[0]+PVFMM_COORD_DIM));
              //memcpy(&nodes[j]->surf_coord[0],& src_coord[0]+part_indx[j]*(           PVFMM_COORD_DIM),n_pts*(           PVFMM_COORD_DIM)*sizeof(Real));
              memcpy(nodes[j]->surf_value.begin(),surf_value.begin()+part_indx[j]*(ker_dim[0]+PVFMM_COORD_DIM),n_pts*(ker_dim[0]+PVFMM_COORD_DIM)*sizeof(Real));
            }
          }
        }
      }
    }
    { // Set trg tree_data
      if(trg_pos==src_pos && n_src==n_trg){ // Scatter trg data
        trg_coord.ReInit(src_coord.Dim(),src_coord.begin(),false);
      }else{
        // Compute MortonId and copy coordinates.
        if((size_t)trg_coord.Dim()!=(size_t)(n_trg*PVFMM_COORD_DIM)) trg_coord.ReInit(n_trg*PVFMM_COORD_DIM);
        pt_mid    .ReInit(n_trg);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          size_t a=((tid+0)*n_trg)/omp_p;
          size_t b=((tid+1)*n_trg)/omp_p;
          for(size_t i=a;i<b;i++){
            for(size_t j=0;j<PVFMM_COORD_DIM;j++){
              trg_coord[i*PVFMM_COORD_DIM+j]=trg_pos[i*PVFMM_COORD_DIM+j]*scale_x+shift_x[j];
              while(trg_coord[i*PVFMM_COORD_DIM+j]< 0.0) trg_coord[i*PVFMM_COORD_DIM+j]+=1;
              while(trg_coord[i*PVFMM_COORD_DIM+j]>=1.0) trg_coord[i*PVFMM_COORD_DIM+j]-=1;
            }
            pt_mid[i]=pvfmm::MortonId(sctl::Ptr2ConstItr<Real>(&trg_coord[i*PVFMM_COORD_DIM],3));
          }
        }

        // Scatter trg coordinates.
        ctx->sctl_comm.SortScatterIndex( pt_mid  , scatter_index, &min_mid);
        ctx->sctl_comm.ScatterForward( pt_mid  , scatter_index);
        ctx->sctl_comm.ScatterForward(trg_coord, scatter_index);
      }
      { // Set trg tree_data
        std::vector<size_t> part_indx(nodes.size()+1);
        part_indx[nodes.size()]=pt_mid.Dim();
        #pragma omp parallel for
        for(size_t j=0;j<nodes.size();j++){
          part_indx[j]=std::lower_bound(pt_mid.begin(), pt_mid.begin()+pt_mid.Dim(), nodes[j]->GetMortonId())-pt_mid.begin();
        }

        if(setup){
          #pragma omp parallel for
          for(size_t j=0;j<nodes.size();j++){
            size_t n_pts=part_indx[j+1]-part_indx[j];
            {
              nodes[j]-> trg_coord.ReInit(n_pts*(PVFMM_COORD_DIM), trg_coord.begin()+part_indx[j]*(PVFMM_COORD_DIM),false);
            }
          }
        }else{
          #pragma omp parallel for
          for(size_t j=0;j<nodes.size();j++){
            size_t n_pts=part_indx[j+1]-part_indx[j];
            {
              assert((size_t)nodes[j]->trg_coord.Dim()==n_pts*(PVFMM_COORD_DIM));
              //memcpy(&nodes[j]->trg_coord[0],&trg_coord[0]+part_indx[j]*(PVFMM_COORD_DIM),n_pts*(PVFMM_COORD_DIM)*sizeof(Real));
            }
          }
        }
      }
    }
  }

  if(setup){ // Optional stuff (redistribute, adaptive refine ...)
    auto print_tree_stats = [](void* ctx_) { //Output max tree depth.
      typedef pvfmm::FMM_Node<pvfmm::MPI_Node<Real> > Node_t;
      //typedef pvfmm::FMM_Pts<Node_t> Mat_t;
      //typedef pvfmm::FMM_Tree<Mat_t> Tree_t;

      PVFMMContext<Real>* ctx=(PVFMMContext<Real>*)ctx_;

      int myrank;
      myrank = ctx->sctl_comm.Rank();

      long nleaf=0, maxdepth=0;
      std::vector<size_t> all_nodes(sctl::MAX_DEPTH+1,0);
      std::vector<size_t> leaf_nodes(sctl::MAX_DEPTH+1,0);
      std::vector<sctl::Iterator<Node_t>>& nodes=ctx->tree->GetNodeList();
      for(size_t i=0;i<nodes.size();i++){
        sctl::Iterator<Node_t> n=nodes[i];
        if(!n->IsGhost()) all_nodes[n->Depth()]++;
        if(!n->IsGhost() && n->IsLeaf()){
          leaf_nodes[n->Depth()]++;
          if(maxdepth<n->Depth()) maxdepth=n->Depth();
          nleaf++;
        }
      }

      std::stringstream os1,os2;
      os1<<"All Nodes";
      for(int i=0;i<sctl::MAX_DEPTH;i++){
        int local_size=all_nodes[i];
        int global_size;
        ctx->sctl_comm.Allreduce(sctl::Ptr2ConstItr<int>(&local_size,1), sctl::Ptr2Itr<int>(&global_size,1), 1, sctl::CommOp::SUM);
        os1<<global_size<<' ';
      }
      if(!myrank) std::cout<<os1.str()<<'\n';

      os2<<"Leaf Nodes: ";
      for(int i=0;i<sctl::MAX_DEPTH;i++){
        int local_size=leaf_nodes[i];
        int global_size;
        ctx->sctl_comm.Allreduce(sctl::Ptr2ConstItr<int>(&local_size,1), sctl::Ptr2Itr<int>(&global_size,1), 1, sctl::CommOp::SUM);
        os2<<global_size<<' ';
      }
      if(!myrank) std::cout<<os2.str()<<'\n';

      long nleaf_glb=0, maxdepth_glb=0;
      { // MPI_Reduce
        ctx->sctl_comm.Allreduce(sctl::Ptr2ConstItr<int>(&nleaf,1), sctl::Ptr2Itr<int>((int*)&nleaf_glb,1), 1, sctl::CommOp::SUM);
        ctx->sctl_comm.Allreduce(sctl::Ptr2ConstItr<int>(&maxdepth,1), sctl::Ptr2Itr<int>((int*)&maxdepth_glb,1), 1, sctl::CommOp::MAX);
      }
      if(!myrank) std::cout<<"Number of Leaf Nodes: "<<nleaf_glb<<'\n';
      if(!myrank) std::cout<<"Tree Depth: "<<maxdepth_glb<<'\n';
    };
    PVFMM_UNUSED(print_tree_stats);
    //print_tree_stats<Real>(ctx_);
    ctx->tree->InitFMM_Tree(true, ctx->bndry);
    //print_tree_stats<Real>(ctx_);
  }

  // Setup tree for FMM.
  if(setup) ctx->tree->SetupFMM(ctx->mat);
  else ctx->tree->ClearFMMData();
  ctx->tree->RunFMM();

  { // Get target potential.
    sctl::Vector<Real> trg_value;
    { // Get trg data.
      sctl::Iterator<Node_t> n=ctx->tree->PreorderFirst();
      while(n!=sctl::NullIterator<Node_t>()){
        if(!n->IsGhost() && n->IsLeaf()) break;
        n=ctx->tree->PreorderNxt(n);
      }
      assert(n!=sctl::NullIterator<Node_t>());

      size_t trg_size=0;
      const std::vector<sctl::Iterator<Node_t>>& nodes=ctx->tree->GetNodeList();
      #pragma omp parallel for reduction(+:trg_size)
      for(size_t i=0;i<nodes.size();i++){
        if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
          trg_size+=nodes[i]->trg_value.Dim();
        }
      }
      trg_value.ReInit(trg_size,n->trg_value.begin());
    }
    ctx->sctl_comm.ScatterReverse(trg_value, scatter_index, n_trg);
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      size_t a=((tid+0)*n_trg)/omp_p;
      size_t b=((tid+1)*n_trg)/omp_p;
      for(size_t i=a;i<b;i++){
        for(int k=0;k<ker_dim[1];k++){
          trg_val[i*ker_dim[1]+k]=trg_value[i*ker_dim[1]+k]*trg_scal[k];
        }
      }
    }
  }
  sctl::Profile::Toc();
}

template<typename Real> static void PVFMMDestroyContext(void** ctx){
  if(!ctx[0]) return;

  // Delete tree.
  delete ((PVFMMContext<Real>*)ctx[0])->tree;

  // Delete matrices.
  delete ((PVFMMContext<Real>*)ctx[0])->mat;

  // Delete context.
  delete (PVFMMContext<Real>*)ctx[0];
  ctx[0]=NULL;
}

#ifdef __cplusplus
extern "C" {
#endif

// Evaluate potential in single-precision
void pvfmmevalf_(const float* src_pos, const float* sl_den, int64_t* n_src, const float* trg_pos, float* trg_val, int64_t* n_trg, void** ctx_, int32_t* setup) {
  PVFMMEval<float>(src_pos, sl_den, nullptr, *n_src, trg_pos, trg_val, *n_trg, *ctx_, *setup);
}

// Create single-precision particle FMM context
void pvfmmcreatecontextf_(void** ctx, float* box_size, int32_t* n, int32_t* m, int32_t* kernel, int32_t* bndry, MPI_Fint* fcomm) {
  MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  (*ctx) = PVFMMCreateContext<float>(*box_size, *n, *m, sctl::MAX_DEPTH, PVFMMKernelPtr<float>((enum PVFMMKernel)*kernel), PVFMMBndry(*bndry), comm);
}

// Destroy single-precision particle FMM context
void pvfmmdestroycontextf_(void** ctx) {
  PVFMMDestroyContext<float>(ctx);
}


// Evaluate potential in double-precision
void pvfmmevald_(const double* src_pos, const double* sl_den, int64_t* n_src, const double* trg_pos, double* trg_val, int64_t* n_trg, void** ctx_, int32_t* setup) {
  PVFMMEval<double>(src_pos, sl_den, nullptr, *n_src, trg_pos, trg_val, *n_trg, *ctx_, *setup);
}

// Create double-precision particle FMM context
void pvfmmcreatecontextd_(void** ctx, double* box_size, int32_t* n, int32_t* m, int32_t* kernel, int32_t* bndry, MPI_Fint* fcomm) {
  MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  (*ctx) = PVFMMCreateContext<double>(*box_size, *n, *m, sctl::MAX_DEPTH, PVFMMKernelPtr<double>((enum PVFMMKernel)*kernel), PVFMMBndry(*bndry), comm);
}

// Destroy double-precision particle FMM context
void pvfmmdestroycontextd_(void** ctx) {
  PVFMMDestroyContext<double>(ctx);
}




void* PVFMMCreateContextF(float box_size, int n, int m, enum PVFMMKernel kernel, enum PVFMMBoundaryType bndry, MPI_Comm comm) {
  return PVFMMCreateContext<float>(box_size, n, m, sctl::MAX_DEPTH, PVFMMKernelPtr<float>(kernel), PVFMMBndry(bndry), comm);
}

void PVFMMEvalF(const float* src_pos, const float* sl_den, const float* dl_den, long n_src, const float* trg_pos, float* trg_val, long n_trg, const void* ctx, int setup) {
  PVFMMEval<float>(src_pos, sl_den, dl_den, n_src, trg_pos, trg_val, n_trg, ctx, setup);
}

void PVFMMDestroyContextF(void** ctx) {
  PVFMMDestroyContext<float>(ctx);
}


void* PVFMMCreateContextD(double box_size, int n, int m, enum PVFMMKernel kernel, enum PVFMMBoundaryType bndry, MPI_Comm comm) {
  return PVFMMCreateContext<double>(box_size, n, m, sctl::MAX_DEPTH, PVFMMKernelPtr<double>(kernel), PVFMMBndry(bndry), comm);
}

void PVFMMEvalD(const double* src_pos, const double* sl_den, const double* dl_den, long n_src, const double* trg_pos, double* trg_val, long n_trg, const void* ctx, int setup) {
  PVFMMEval<double>(src_pos, sl_den, dl_den, n_src, trg_pos, trg_val, n_trg, ctx, setup);
}

void PVFMMDestroyContextD(void** ctx) {
  PVFMMDestroyContext<double>(ctx);
}

MPI_Fint PVFMMGetCommWorld(void) {
  PVFMMEnsureMPIInitialized();
#if defined(SCTL_HAVE_MPI)
  return MPI_Comm_c2f(MPI_COMM_WORLD);
#else
  return 0;  // ignored by the no-MPI build (single-process / self-comm)
#endif
}


#ifdef __cplusplus
}
#endif
