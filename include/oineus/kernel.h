#include <vector>
#include <unordered_map>
#include <cmath>
#include "simplex.h"
#include "sparse_matrix.h"
#include <numeric>
#include <future>
#include <oneapi/tbb/parallel_for.h>
// suppress pragma message from boost
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#pragma once

namespace oineus {
	template<typename Int_, typename Real_>
	struct KerImCokReduced {
		using Int = Int_;
		using Real = Real_;
        using IntSparseColumn = SparseColumn<Int>;
        using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
    	using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;
		using Point = DgmPoint<Real>;
    	//using Dgm = oineus::Diagram;
		using Dgms = oineus::Diagrams<Real>;

		private:
			Filtration<Int_, Real_, Int_> K; //Full complex with the function values for F
			Filtration<Int_, Real_, Int_> L; //Sub complex with the function values for G
			VRUDecomp F; //the reduced triple for F0
			VRUDecomp G; //reduced triple for G
			VRUDecomp Im; //reduced image triple
			VRUDecomp Ker; //reduced kernel triple
			VRUDecomp Cok; //reduced cokernel triple
			Dgms KerDiagrams; //vector of image diagrams, one in each dimension poissble (these may be empty)
			Dgms ImDiagrams; //vector of kernel diagrams, one in each dimension poissble (these may be empty)
			Dgms CokDiagrams; //vector of kernel diagramsm, one in each possible dimension (these may be empty)
			int max_dim; //the maximum dimension of a cell
			std::vector<int> sorted_K_to_sorted_L;
			std::vector<int> sorted_L_to_sorted_K;
			int number_cells_K; //number of cells in K
			int number_cells_L; //number of cells in L
			std::vector<int> new_order_to_old; //new_order_to_old[i] is the (unsorted) id in K of the ith cell in the filtration.
			std::vector<int> new_cols; //the id of the columns we retain
			Params params;

		public:
			//std::vector<bool> to_keep;
			//Constructor which takes as input the complex K, subcomplex L, and the decompositionfs for F, G, Im, Ker, Cok, as well as the map from sorted L to sorted K and sorted K to sorted L, as well as the change in ordering to have L before K.
			KerImCokReduced(Filtration<Int_, Real_, Int_> K_, Filtration<Int_, Real_, Int_> L_,VRUDecomp F_, VRUDecomp G_, VRUDecomp Im_, VRUDecomp Ker_, VRUDecomp Cok_, std::vector<int> sorted_L_to_sorted_K_, std::vector<int> sorted_K_to_sorted_L_, std::vector<int> new_order_to_old_,std::vector<int> new_cols_, Params params_) :
				K (K_),
				L (L_),
				F (F_),
				G (G_),
				Im (Im_),
				Ker (Ker_),
				Cok (Cok_),
				sorted_L_to_sorted_K (sorted_L_to_sorted_K_),
				sorted_K_to_sorted_L (sorted_K_to_sorted_L_),
				new_order_to_old (new_order_to_old_),
				new_cols (new_cols_),
				params (params_) {
					number_cells_K = K.boundary_matrix_full().size(); //set the number of cells in K
					number_cells_L = L.boundary_matrix_full().size(); //set the number of cells in L
					max_dim = K.max_dim(); //set the maximal dimension we can have cycles in. 
					Dgms KerDiagrams(max_dim+1);
					Dgms ImDiagrams(max_dim+1);
					Dgms CokDiagrams(max_dim+1);
					//std::vector<bool> to_keep(number_cells_K, false);
				}
			
			

			//generate the kernel persistence diagrams and store them in KerDiagrams
			void GenerateKerDiagrams() {//Extract the points in the image diagrams.
				std::cerr << "Generating kernel diagrams." << std::endl;

				std::vector<bool> open_points_ker (number_cells_K); //keep track of points which give birth to a cycle

				for (int i = 0; i < number_cells_K; i++) {
                    if (sorted_K_to_sorted_L[i] == -1) {//To give birth the cell must be is in K-L
                        if (!F.get_R()[i].empty()) {//Now check negativitiy in F, which requires the column to be non-zero, which given we are working with column sparse binary matrices is equivalent to the column being non-empty
                            bool cycle_f = false; //V[i] can only store a cycle if it is non-zero
							if (!F.get_V()[i].empty()) {
								cycle_f = true; //easier to check if something is not a cycle than if it is, so we assume it is and then check
								std::vector<int> quasi_sum_f(number_cells_K, 0);
								for (int j = 0; j < F.get_V()[i].size(); j++){
									for (int k = 0; k < F.get_D()[F.get_V()[i][j]].size(); k++) {
										quasi_sum_f[F.get_D()[F.get_V()[i][j]][k]]++;
									}
								}
								for (int j = 0; j < quasi_sum_f.size(); j++) {
									if (quasi_sum_f[j] %2 != 0) {//if one entry is not 0 mod 2, then V[i] does not store a cycle
										cycle_f = false;
										break;
									}
								}
							}
                            if (!cycle_f) {
								if (sorted_K_to_sorted_L[Im.get_R()[i].back()] != -1) {//Check lowest 1 in R_im is in L
                                    open_points_ker[i] = true;
                                }
                            }
                        }
                    } else if (F.get_R()[i].empty()) {//to kill something, i needs to be positive in f and negative in g, and so we begin with testing positive in f
                        bool cycle_f = false; //if V[i] is empty, cannot store a cycle, so we start with cycle as false, and then change to true if V[i] is not zero
                        if (!F.get_V()[i].empty()) {//same as previous cycle check
                            cycle_f = true;
                            std::vector<int> quasi_sum_f(number_cells_K, 0);
                            for (int j = 0; j < F.get_V()[i].size(); j++){
                                for (int k = 0; k < F.get_D()[F.get_V()[i][j]].size(); k++) {
                                    quasi_sum_f[F.get_D()[F.get_V()[i][j]][k]]++;
                                }
                            }
                            for (int j = 0; j < quasi_sum_f.size(); j++) {//if there is an entry in quasi_sum that is not 0 mod 2 then does not represent a cycle
                                if (quasi_sum_f[j] %2 != 0) {
                                    cycle_f = false;
                                    break;
                                }
                            }
                        }
                        if (cycle_f) {
                            if (!G.get_R()[sorted_K_to_sorted_L[i]].empty()) {//check if negative in G
                                bool cycle_g = false;
								if (!G.get_V()[i].empty()){ //same as previous cycle checks
									cycle_g = true;
									std::vector<int> quasi_sum_g(number_cells_L, 0);
									for (int j = 0; j < G.get_V()[sorted_K_to_sorted_L[i]].size(); j++){
										for (int k = 0; k < G.get_D()[G.get_V()[sorted_K_to_sorted_L[i]][j]].size(); k++) {
											quasi_sum_g[G.get_D()[G.get_V()[sorted_K_to_sorted_L[i]][j]][k]]++;
										}
									}
									for (int j = 0; j < quasi_sum_g.size(); j++) {
										if (quasi_sum_g[j] %2 != 0) {
											cycle_g = false;
											break;
										}
									}
								}
                                if (!cycle_g) {
                                    int birth_id = new_order_to_old[Ker.get_R()[new_cols[i]].back()];
                                    int dim = K.dim_by_sorted_id(i)-1;
                                    if (K.value_by_sorted_id(birth_id) != K.value_by_sorted_id(i)) {
                                        KerDiagrams.add_point(dim, K.value_by_sorted_id(birth_id), K.value_by_sorted_id(i));//[dim].push_back(Point(K.value_by_sorted_id(birth_id), K.value_by_sorted_id(i))); //add point to the diagram
                                        open_points_ker[birth_id] = false; //close the point which gave birth to the cycle that was just killed, so we don't add an point at inifity to the diagram
                                }
                            }
                        }
                    }
                }

				for (int i = 0; i < open_points_ker.size(); i++) {//check if there are any cycles with infinite life times
					if (open_points_ker[i]) {
						int dim = K.dim_by_sorted_id(i)-1;
						KerDiagrams.add_point(dim,K.value_by_sorted_id(i), std::numeric_limits<double>::infinity()); //add point to the diagram
					}
				}
				
				std::cerr << "The kernel diagrams are: " << std::endl;
				for (int i = 0; i <= max_dim; i++) {
					if (KerDiagrams.extract(i).empty()) {
						std::cerr << "Diagram in dimension " << i << " is empty." << std::endl;
					} else {
						std::cerr << "Diagram in dimension " << i << " is:" << std::endl;
						for (int j = 0; j < KerDiagrams[i].size(); j++) {
							std::cerr << "(" << KerDiagrams[i][j].birth << ", " << KerDiagrams[i][j].death << ")" << std::endl;
						}
					}
				}
			}

			//generate the image persistence diagrams and store them in ImDiagrams
			void GenerateImDiagrams() {
				std::cerr << "Generating the image diagrams." << std::endl;

				std::vector<bool> open_points_im (number_cells_K);//keep track of cells which give birth to a cycle and if that cycle is killed or nop

				for (int i = 0; i < number_cells_K; i++){
                    if (sorted_K_to_sorted_L[i] != -1) {//a cell only gives birth if it is in L

                        int sorted_id_in_L = sorted_K_to_sorted_L[i];//we are now working with a cell in L, and checking properties only L with respect to G, so we need to use the sorted id of the cell in L
                        if (G.get_R()[sorted_id_in_L].empty()) {//needs to be positive in G
                            std::vector<int> quasi_sum_g(number_cells_L, 0);
                            bool cycle_g = false; //same as previous cycle checks.
                            if (!G.get_V()[sorted_id_in_L].empty()){
                                cycle_g = true;
                                for (int j = 0; j < G.get_V()[sorted_id_in_L].size(); j++) {
                                    for (int k = 0; k < G.get_D()[G.get_V()[sorted_id_in_L][j]].size(); k++) {
                                        quasi_sum_g[G.get_D()[G.get_V()[sorted_id_in_L][j]][k]]++;
                                    }
                                }
                                for (int k = 0; k < quasi_sum_g.size(); k++) {
                                    if (quasi_sum_g[k] %2 != 0) {
                                        cycle_g = false;
                                        break;
                                    }
                                }
                            }
                            if (cycle_g) {
                                    open_points_im[i] = true;
							}
                        }
                    }
                    //any cell can kill something, so we just test negativitity in F
                    if (!F.get_R()[i].empty()) {//F.get_R()[i] needs to be non-zero, which due to binary sparse format is the same as being non-empty
						bool cycle_f = false;
                        if (!F.get_V()[i].empty()) {
                            cycle_f = true;
                            std::vector<int> quasi_sum_f(number_cells_K, 0);
                            for (int j = 0; j < F.get_V()[i].size(); j++) {
                                for (int k = 0; k < F.get_D()[F.get_V()[i][j]].size(); k++){
                                    quasi_sum_f[F.get_D()[F.get_V()[i][j]][k]]++;
                                }                           }
                            for (int k = 0; k < quasi_sum_f.size(); k++) {
                                if (quasi_sum_f[k] %2 != 0) {
                                    cycle_f = false;
                                    break;
                                }
                            }
                        }
                        if (!cycle_f) {
							int birth_id = new_order_to_old[Im.get_R()[i].back()];
							if (sorted_K_to_sorted_L[birth_id] != -1) {
								int dim = K.dim_by_sorted_id(i)-1;
								if (K.value_by_sorted_id(birth_id) != K.value_by_sorted_id(i)) {
									ImDiagrams.add_point(dim, K.value_by_sorted_id(birth_id), K.value_by_sorted_id(i));//add the point to the diaram
								}
								open_points_im[birth_id] = false;
							}
                        }
                    }
                }
				//for any cycle with infinite life, add the point to the diagram
				for (int i = 0; i < open_points_im.size(); i++) {
					if (open_points_im[i]) {
						int dim = K.dim_by_sorted_id(i);
						ImDiagrams.add_point(dim, K.value_by_sorted_id(i), std::numeric_limits<double>::infinity());
					}
				}

				std::cerr << "The image diagrams are: " << std::endl;
				for (int i = 0; i <= max_dim; i++) {
					if (ImDiagrams.extract(i).empty()) {
						std::cerr << "Diagram in dimension " << i << " is empty." << std::endl;
					} else if (!ImDiagrams.extract(i).empty()) {
						std::cerr << "Diagram in dimension " << i << " is:" << std::endl;
						for (int j = 0; j < ImDiagrams[i].size(); j++) {
							std::cerr << "(" << ImDiagrams[i][j].birth << ", " << ImDiagrams[i][j].death << ")" << std::endl;
						}
					}
				}
			}
			
			//Generate the cokernel diagrams and store them in CokDiagrams
			void GenerateCokDiagrams() {
				std::cerr << "Starting to extract the cokernel diagrams." << std::endl;

				std::vector<bool> open_points_cok (number_cells_K);//keep track of open cycles

				for (int i = 0; i < number_cells_K; i++) {//we first check that i is positive in f
					if (F.get_R()[i].empty()) {//the column in R needs to be empty
						bool cycle_f = false;
						if (!F.get_V()[i].empty()) {//standard cycle check as previous ones
							cycle_f = true;
							std::vector<int> quasi_sum_f(number_cells_K, 0);
							for (int j = 0; j < F.get_V()[i].size(); j++) {
								for (int k = 0; k < F.get_D()[F.get_V()[i][j]].size(); k++) {
									quasi_sum_f[F.get_D()[F.get_V()[i][j]][k]]++;
								}
							}
							for (int k = 0; k < quasi_sum_f.size(); k++) {
								if (quasi_sum_f[k] %2 != 0) {
									cycle_f = false;
									break;
								}
							}
						}
						if (cycle_f) {
							//now we check if either i is in K-L or negative in g
							if (sorted_K_to_sorted_L[i] == -1) {
								open_points_cok[i] = true;
							} else {
								if (!G.get_R()[sorted_K_to_sorted_L[i]].empty()) {
									bool cycle_g = false;
									if (!G.get_V()[sorted_K_to_sorted_L[i]].empty()) {//standard cycle check
										cycle_g= true;
										std::vector<int> quasi_sum_g(number_cells_L, 0);
										for (int j = 0; j < G.get_V()[sorted_K_to_sorted_L[i]].size(); j++) {
											for (int k = 0; k < G.get_D()[G.get_V()[sorted_K_to_sorted_L[i]][j]].size(); k++) {
												quasi_sum_g[G.get_D()[G.get_V()[sorted_K_to_sorted_L[i]][j]][k]]++;
											}
										}
										for (int k = 0; k < quasi_sum_g.size(); k++) {
											if (quasi_sum_g[k] %2 != 0) {
												cycle_g = false;
												break;
											}
										}
									}
									if (!cycle_g) {
										open_points_cok[i] = true;
									}
								}
							}
						}
					} else if (!Im.get_V()[i].empty()){
						bool cycle_im = false;
						if (!Im.get_V()[i].empty()) {
							cycle_im = true;
							std::vector<int> quasi_sum_f(number_cells_K, 0);
							for (int j = 0; j < Im.get_V()[i].size(); j++) {
								for (int k = 0; k < Im.get_D()[Im.get_V()[i][j]].size(); k++) {
									quasi_sum_f[Im.get_D()[Im.get_V()[i][j]][k]]++;
								}
							}
							for (int k = 0; k < quasi_sum_f.size(); k++) {
								if (quasi_sum_f[k] %2 != 0) {
									cycle_im = false;
									break;
								}
							}
						}
						if (!cycle_im && sorted_K_to_sorted_L[new_order_to_old[Im.get_R()[i].back()]] == -1) {
							int birth_id = new_order_to_old[Cok.get_R()[i].back()];
							if (K.value_by_sorted_id(birth_id) != K.value_by_sorted_id(i)) {
								int dim = K.dim_by_sorted_id(birth_id);
								CokDiagrams.add_point(dim, K.value_by_sorted_id(birth_id), K.value_by_sorted_id(i));//add point to the diagram
							}
							open_points_cok[birth_id] = false;
						}
					}
				}

				//for any open cycles add a point with infinite lifetime to the diagram
				for (int i = 0; i < open_points_cok.size(); i++) {
					if (open_points_cok[i]) {
						int dim = K.dim_by_sorted_id(i);
						CokDiagrams.add_point(dim, K.value_by_sorted_id(i), std::numeric_limits<double>::infinity());
					}
				}

				std::cerr << "The cokernel diagrams are: " << std::endl;
				for (int i = 0; i <= max_dim; i++) {
					if (CokDiagrams.extract(i).empty()) {
						std::cerr << "Diagram in dimension " << i << " is empty." << std::endl;
					} else if (!CokDiagrams.extract(i).empty()) {
						std::cerr << "Diagram in dimension " << i << " is:" << std::endl;
						for (int j = 0; j < CokDiagrams[i].size(); j++) {
							std::cerr << "(" << CokDiagrams[i][j].birth << ", " << CokDiagrams[i][j].death << ")" << std::endl;
						}
					}
				}
			}

			//Useful functions to obtain the various matrices. Mostly useful in debugging, but potentially useful for other people depending on applications.
			MatrixData get_D_f() {
				return F.get_D();
			}

			MatrixData get_V_f() {
				return F.get_V();
			}

			MatrixData get_R_f() {
				return F.get_R();
			}

			MatrixData get_D_g() {
				return G.get_D();
			}

			MatrixData get_V_g() {
				return G.get_V();
			}

			MatrixData get_R_g() {
				return G.get_R();
			}

			MatrixData get_D_im() {
				return Im.get_D();
			}

			MatrixData get_V_im() {
				return Im.get_V();
			}

			MatrixData get_R_im() {
				return Im.get_R();
			}

			MatrixData get_D_ker() {
				return Ker.get_D();
			}

			MatrixData get_V_ker() {
				return Ker.get_V();
			}

			MatrixData get_R_ker() {
				return Ker.get_R();
			}
			
			MatrixData get_D_cok() {
				return Cok.get_D();
			}

			MatrixData get_V_cok() {
				return Cok.get_V();
			}

			MatrixData get_R_cok() {
				return Cok.get_R();
			}
			Dgms get_kernel_diagrams(){
				return KerDiagrams;
			}
	
			Dgms get_image_diagrams(){
				return ImDiagrams;
			}

			Dgms get_cokernel_diagrams(){
				return CokDiagrams;
			}
	};

	//Function which takes as input a complex K, a subcomplex L (only requirement is sorted by dimension), and a map from L to K, as well as params,te
	template <typename Int_, typename Real_>
	KerImCokReduced<Int_, Real_> reduce_ker_im_cok(Filtration<Int_, Real_, Int_> K, Filtration<Int_, Real_, Int_> L, std::vector<int> L_to_K, Params& params) {
		using Real = Real_;
		using Int = Int_;
    	using IntSparseColumn = SparseColumn<Int>;
    	using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
   		using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;
		using Point = DgmPoint<Real>;
		using Diagram = std::vector<Point>;

		std::cerr << "Performing kernel, image, cokernel reduction with the following parameters:" << std::endl;
		std::cerr << "n_threads: " << params.n_threads << std::endl;
		std::cerr << "kernel: " << params.kernel << std::endl;
		std::cerr << "image: " << params.image << std::endl;
		std::cerr << "cokernel: " << params.cokernel << std::endl;
		std::cerr << "verbose: " << params.verbose << std::endl;

		FiltrationSimplexVector K_simps = K.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_K =  K_simps.size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_L =  L_simps.size(); // number of simplices in L

		std::vector<int> sorted_L_to_sorted_K(number_cells_L, 0); //need to create the map from sorted L to sorted K
		std::vector<int> sorted_K_to_sorted_L(number_cells_K, -1); //need a map from sorted K to sorted L, for any cell not in L, we set the value to -1, which is convenient for getting the diagrams.	


		for (int i = 0; i < number_cells_L; i++) {//getting sorted L to sorted K is relatively easy
			sorted_L_to_sorted_K[L.get_sorted_id(i)] = K.get_sorted_id(L_to_K[i]);
		}

		for (int i = 0; i < number_cells_L; i++) {//for cells in K which are also in L, set the sorted id, which we can get from sorted L to sorted K
			sorted_K_to_sorted_L[sorted_L_to_sorted_K[i]] = i;
		}

		//set up the reduction for F  on K
		if (params.verbose) std::cerr << "Reducing F on K." << std::endl;
		VRUDecomp F(K.boundary_matrix_full());
		F.reduce_parallel_rvu(params);

		//set up reduction for G on L
		if (params.verbose) std::cerr << "Reducing G on L." << std::endl;
		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);

		std::vector<int> new_order (number_cells_K);//we will need to reorder rows so that L comes first and then K-L
		std::iota (new_order.begin(), new_order.end(), 0);

		if (params.verbose) std::cerr << "Sorting so that cells in L come before cells in K." << std::endl;
		std::sort(new_order.begin(), new_order.end(), [&](int i, int j) {//sort so that all cells in L come first sorted by dimension and then value in G, and then cells in K-L sorted by dimension and value in F
			if (sorted_K_to_sorted_L[i] != -1 && sorted_K_to_sorted_L[j] != -1) {//if both are in L, sort by dimension and then value under G
				int i_dim, j_dim;
				double i_val, j_val;
				i_dim = L.dim_by_sorted_id(sorted_K_to_sorted_L[i]);
				j_dim = L.dim_by_sorted_id(sorted_K_to_sorted_L[j]);
				if (i_dim == j_dim) {
					i_val = L.value_by_sorted_id(sorted_K_to_sorted_L[i]);
					j_val = L.value_by_sorted_id(sorted_K_to_sorted_L[j]);
					if (i_val == j_val) {
						if ( i < j) {
							return true;
						} else {
							return false;
						}
					} else if  (i_val < j_val) {
						return true;
					} else {
						return false;
					}
				} else {
					if (i_dim < j_dim) {
						return true;
					} else {
						return false;
					}
				}
			} else if (sorted_K_to_sorted_L[i] != -1 && sorted_K_to_sorted_L[j] == -1) {//i is in L and j is not
				return true;
			} else if (sorted_K_to_sorted_L[i] == -1 && sorted_K_to_sorted_L[j] != -1) {//i is not in L but j is
				return false;
			} else {
				int i_dim, j_dim;
				double i_val, j_val;
				i_dim = K.dim_by_sorted_id(i);
				j_dim = K.dim_by_sorted_id(j);
				if (i_dim == j_dim) {
					i_val = K.value_by_sorted_id(i);
					j_val = K.value_by_sorted_id(j);
					if (i_val == j_val) {
						if ( i < j) {
							return true;
						} else {
							return false;
						}
					} else if  (i_val < j_val) {
						return true;
					} else {
						return false;
					}
				} else {
					if (i_dim < j_dim) {
						return true;
					} else {
						return false;
					}
				}
			}
		});

		std::vector<int> old_to_new_order(number_cells_K);//map from old order to new order so that we know which cells correspond to which rows. This could be done by just shuffling the row indices, but as we create a new reduction isntance, we need to create a new matrix anyway.

		for (int i = 0; i < number_cells_K; i++) {
			old_to_new_order[new_order[i]] = i;
		}

		MatrixData d_im;
		for (int i = 0; i < F.get_D().size(); i++) {
			std::vector<int> new_col_i;
			if (!F.get_D()[i].empty()) {
				for (int j = 0; j < F.get_D()[i].size(); j++) {
					new_col_i.push_back(old_to_new_order[F.get_D()[i][j]]);
				}
			}
			std::sort(new_col_i.begin(), new_col_i.end());//sort to make sure this is all correct. 
			d_im.push_back(new_col_i);
		}


        params.clearing_opt = false;//set clearing to false as this was interferring with the change in row order
		//set up Im reduction
		if (params.verbose) std::cerr << "Reducing Image." << std::endl;
		VRUDecomp Im(d_im);
		Im.reduce_parallel_rvu(params); 

		//we need to remove some columns from Im to get Ker, so we need to know which ones we keep, and then what cells they correspond t
		if (params.verbose) std::cerr << "Checking which columns to keep." << std::endl;
		std::vector<char> to_keep(number_cells_K);
		//using MatrixData = std::vector<std::vector<int> >;

		const MatrixData F_V(F.get_V());
		const MatrixData F_D(F.get_D());
		tbb::parallel_for(tbb::blocked_range<std::size_t>(0, number_cells_K), [&](const tbb::blocked_range<std::size_t> &r) {
			for (int i=r.begin(); i < r.end(); i++){
				if (!F_V[i].empty()) {//cycle check as in the code for generating the persistence diagrams.
					bool cycle = true;
					std::vector<int> quasi_sum (number_cells_K, 0);
					for (int j = 0; j < F_V[i].size(); j++) {
						for (int k = 0; k < F_D[F_V[i][j]].size(); k++) {
							quasi_sum[F_D[F_V[i][j]][k]]++;
						}
					}
					for (int j = 0; j < quasi_sum.size(); j++) {
						if (quasi_sum[j]%2 != 0) {
							break;
						}
					}
					if (cycle) {
						to_keep[i] = 't';
					} else {
						to_keep[i] = 'f';
					}
				};
			};
		});
		
		MatrixData d_ker;
		std::vector<int> new_cols(number_cells_K, -1);
		int counter = 0;
		for (int i = 0; i < number_cells_K; i++) {
			if (to_keep[i] == 't') {
				d_ker.push_back(Im.get_V()[i]);
				new_cols[i] = counter;
				counter++;
			}
		}
		if (params.verbose) std::cerr << "Reducing Ker." << std::endl;
		VRUDecomp Ker(d_ker, K.size());
		Ker.reduce_parallel_rvu(params);
		MatrixData d_cok(Im.get_D());


		for (int i = 0; i < number_cells_L; i++) {
			bool replace = false;
			std::vector<int> quasi_sum (number_cells_L, 0);
			if (!G.get_V()[i].empty()) {
				replace = true;
				for (int j = 0; j < G.get_V()[i].size(); j++) {
					for (int k = 0; k < G.get_D()[G.get_V()[i][j]].size(); k++) {
						quasi_sum[G.get_D()[G.get_V()[i][j]][k]]++;//check if a column in V_g represents a cycle
					}
				}
			}
			for (int j = 0; j < quasi_sum.size(); j++) {
				if (quasi_sum[j]%2 !=0) {
					replace = false;
					break;
				}
			}
			if (replace) {
				d_cok[sorted_L_to_sorted_K[i]] = G.get_D()[i];
			}
		}

		if (params.verbose) std::cerr << "Reducing Cok." << std::endl;		
		VRUDecomp Cok(d_cok);
		Cok.reduce_parallel_rvu(params);

		KerImCokReduced<Int, Real> KICR(K, L, F, G, Im, Ker, Cok, sorted_L_to_sorted_K, sorted_K_to_sorted_L, new_order, new_cols, params);

		if (params.kernel) KICR.GenerateKerDiagrams();
		if (params.image) KICR.GenerateImDiagrams();
		if (params.cokernel) KICR.GenerateCokDiagrams();

		return  KICR;
	}
} //end namespace
