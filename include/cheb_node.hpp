/**
 * \file cheb_node.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 1-22-2011
 * \brief This is a derived cheb class of MPI_Node.
 */

#include <vector>
#include <cstdlib>
#include <stdint.h>

#include <pvfmm_common.hpp>
#include <tree_node.hpp>
#include <mpi_node.hpp>
#include <vector.hpp>

#ifndef _PVFMM_CHEB_NODE_HPP_
#define _PVFMM_CHEB_NODE_HPP_

namespace pvfmm{

/**
 * \brief
 */
template <class Real_t>
class Cheb_Node: public MPI_Node<Real_t>{

 public:

  typedef void (*fn_ptr)(Real_t* coord, int n, Real_t* out);

  /**
   * \brief Base class for node data. Contains initialization data for the node.
   */
  class NodeData: public MPI_Node<Real_t>::NodeData{

   public:

     Vector<Real_t> cheb_coord; //Chebyshev point samples.
     Vector<Real_t> cheb_value;

     fn_ptr input_fn; // Function pointer.
     int data_dof;    // Dimension of Chebyshev data.
     int cheb_deg;    // Chebyshev degree
     Real_t tol;      // Tolerance for adaptive refinement.
  };

  /**
   * \brief Initialize pointers to NULL.
   */
  Cheb_Node(): MPI_Node<Real_t>(), input_fn(NULL), cheb_deg(0){}

  /**
   * \brief Virtual destructor.
   */
  virtual ~Cheb_Node();

  /**
   * \brief Initialize the node by passing the relevant data.
   */
  virtual void Initialize(TreeNode* parent_, int path2node_, TreeNode::NodeData*);

  /**
   * \brief Returns list of coordinate and value vectors which need to be
   * sorted and partitioned across MPI processes and the scatter index is
   * saved.
   */
  virtual void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                        std::vector<Vector<Real_t>*>& value,
                        std::vector<Vector<size_t>*>& scatter){
    MPI_Node<Real_t>::NodeDataVec(coord, value, scatter);
    coord  .push_back(&cheb_coord  );
    value  .push_back(&cheb_value  );
    scatter.push_back(&cheb_scatter);

    coord  .push_back(       NULL);
    value  .push_back(&cheb_coeff);
    scatter.push_back(       NULL);
  }

  /**
   * \brief Clear node data.
   */
  virtual void ClearData();

  /**
   * \brief Returns the cost of this node. Used for load balancing.
   */
  virtual long long& NodeCost(){return MPI_Node<Real_t>::NodeCost();}

  /**
   * \brief Degree of Chebyshev polynomials used.
   */
  int ChebDeg(){return cheb_deg;}

  /**
   * \brief Error tolerance for adaptive refinement of the Chebyshev Tree.
   */
  Real_t& MaxErr(){return tol;}

  /**
   * \brief Chebyshev coefficients for the source distribution.
   */
  Vector<Real_t>& ChebData(){return cheb_coeff;}

  /**
   * \brief Allocate a new object of the same type (as the derived class) and
   * return a pointer to it type cast as (TreeNode*).
   */
  virtual TreeNode* NewNode(TreeNode* n_=NULL);

  /**
   * \brief Evaluates and returns the subdivision condition for this node.
   * 'true' if node requires further subdivision.
   */
  virtual bool SubdivCond();

  /**
   * \brief Create child nodes and Initialize them.
   */
  virtual void Subdivide();

  /**
   * \brief Truncates the tree i.e. makes this a leaf node.
   */
  virtual void Truncate();

  /**
   * \brief Return degrees of freedom of data.
   */
  int& DataDOF(){return data_dof;}

  /**
   * \brief Pack this node to be transmitted to another process. The node
   * is responsible for allocating and freeing the memory for the actual data.
   */
  virtual PackedData Pack(bool ghost=false, void* buff_ptr=NULL, size_t offset=0);

  /**
   * \brief Initialize the node with data from another process.
   */
  virtual void Unpack(PackedData data, bool own_data=true);

  /**
   * \brief Read source distribution at points on a grid defined by array of x,
   * y and z coordinates.
   */
  virtual void ReadVal(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, Real_t* val, bool show_ghost=true){
    read_val(x,y,z,x.size(),y.size(),z.size(),val,show_ghost);
  }

  /**
   * \brief Append node VTU data to vectors.
   */
  virtual void VTU_Data(std::vector<Real_t>& coord, std::vector<Real_t>& value, std::vector<int32_t>& connect, std::vector<int32_t>& offset, std::vector<uint8_t>& types, int lod=-1);

  /**
   * \brief Compute gradient of the data.
   */
  void Gradient();

  /**
   * \brief Compute divergence of the data.
   */
  void Divergence();

  /**
   * \brief Compute curl of the data.
   */
  void Curl();

  fn_ptr input_fn;
  Vector<Real_t> cheb_coord;   //coordinates of points
  Vector<Real_t> cheb_value;   //value at points
  Vector<size_t> cheb_scatter; //scatter index mapping original data.

 private:

  /**
   * \brief Read source distribution at points on a grid defined by array of x,
   * y and z coordinates.
   */
  void read_val(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, int nx, int ny, int nz, Real_t* val, bool show_ghost=true);

  Real_t tol;
  int cheb_deg;
  int data_dof;
  Vector<Real_t> cheb_coeff;
};

}//end namespace

#include <cheb_node.txx>

#endif //_PVFMM_CHEB_NODE_HPP_

