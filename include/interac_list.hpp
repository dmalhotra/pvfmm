/**
 * \file interac_list.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-11-2012
 * \brief This file contains the definition of the InteracList class.
 * Handles the logic for different interaction lists, and determines the
 * symmetry class for each interaction.
 */

#ifndef _PVFMM_INTERAC_LIST_HPP_
#define _PVFMM_INTERAC_LIST_HPP_

#include <pvfmm_common.hpp>
#include <tree_node.hpp>
#include <precomp_mat.hpp>

namespace pvfmm{

template <class Node_t>
class InteracList{

  typedef typename Node_t::Real_t Real_t;

  public:

    /**
     * \brief Constructor.
     */
    InteracList(){}

    /**
     * \brief Constructor.
     */
    InteracList(unsigned int dim_){
      Initialize(dim_);
    }

    /**
     * \brief Initialize.
     */
    void Initialize(unsigned int dim_, PrecompMat<Real_t>* mat_=NULL);

    /**
     * \brief Number of possible interactions in each list.
     */
    size_t ListCount(Mat_Type t);

    /**
     * \brief Returns the relative octant coordinates for an interaction i of
     * type t.
     */
    int* RelativeCoord(Mat_Type t, size_t i);

    /**
     * \brief Build interaction list for this node.
     */
    std::vector<Node_t*> BuildList(Node_t* n, Mat_Type t);

    /**
     * \brief For an interaction of type t and index i, returns the symmetry
     * class for the same.
     */
    size_t InteracClass(Mat_Type t, size_t i);

    Matrix<Real_t>& ClassMat(int l, Mat_Type type, size_t indx);

    Permutation<Real_t>& Perm_R(int l, Mat_Type type, size_t indx);

    Permutation<Real_t>& Perm_C(int l, Mat_Type type, size_t indx);

  private:

    /**
     * \brief Returns the list of permutations to be applied to the matrix to
     * convert it to its interac_class.
     */
    std::vector<Perm_Type>& PermutList(Mat_Type t, size_t i);

    /**
     * \brief Set relative coordinates of the interacting node in
     * rel_coord[Type][idx][1:3].
     */
    void InitList(int max_r, int min_r, int step, Mat_Type t);

    /**
     * \brief A hash function defined on the relative coordinates of octants.
     */
    int coord_hash(int* c);

    int class_hash(int* c);

    unsigned int dim;                                //Spatial dimension.
    std::vector<Matrix<int> > rel_coord;             //Relative coordinates of interacting octant.
    std::vector<std::vector<int> > hash_lut;         //Lookup table for hash code of relative coordinates.
    std::vector<std::vector<size_t> > interac_class; //The symmetry class corresponding to each interaction.
    std::vector<std::vector<std::vector<Perm_Type> > > perm_list; //Permutation to convert it to it's interac_class.
    PrecompMat<Real_t>* mat;                         //Handles storage of matrices.
    bool use_symmetries;
};

}//end namespace

#include <interac_list.txx>

#endif //_PVFMM_INTERAC_LIST_HPP_

