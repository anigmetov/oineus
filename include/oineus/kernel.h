#include <vector>
#include <unordered_map>
#include <cmath>
#include "simplex.h"
#include "sparse_matrix.h"
#include <numeric>

// suppress pragma message from boost
#define BOOST_BIND_GLOBAL_PLACEHOLDERS


#pragma once



namespace oineus {

	template<typename Int_, typename Real_>
	struct FilteredPair {
		using Int = Int_;
		using Real = Real_;
        using IntSparseColumn = SparseColumn<Int>;
        using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
    	using FiltrationSimplexVector = std::vector<FiltrationSimplex>;

		public:
			Params params;
            Filtration<Int, Real, int> K;
            Filtration<Int, Real, int> L;
            std::vector<int> L_to_K;

            FilteredPair(const Filtration<Int, Real, int> K_, const Filtration<Int, Real, int> L_, std::vector<int> L_to_K_, const Params params_) { //If the ids of simplices in L_ do not match their ids in K_ we need to know what the correspondence is.
                K = K_;
                L = L_;
                L_to_K = L_to_K_;
                params = params_;
            }

			FilteredPair(const Filtration<Int_, Real_, int> K_, const Filtration<Int_, Real_, int> L_, const Params params_) { // If the ids of simplices in L_ agree with the ids of simplices in K_ we don't need an L_to_K as it it just the identity
                K = K_;
                L = L_;
                L_to_K = std::vector<int> (L.simplices().size());
				std::iota (L_to_K.begin(), L_to_K.end(), 0);
                params = params_;
			}
	};

	template<typename Int_, typename Real_>
	struct ImKerReduced {
		using Int = Int_;
		using Real = Real_;
        using IntSparseColumn = SparseColumn<Int>;
        using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
    	using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;
		using Point = DgmPoint<Real>;
    	using Dgm = std::vector<Point>;
		using Dgms = std::vector<Dgm>;

		private:

			Filtration<Int_, Real_, Int_> K;
			Filtration<Int_, Real_, Int_> L;
			VRUDecomp F; //the reduced triple for F0
			VRUDecomp G; //reduced triple for G
			VRUDecomp Im; //reduced image triple
			VRUDecomp Ker; //reduced kernel triple
			Dgms ImDiagrams; //vector of image diagrams, one in each dimension poissble (these may be empty)
			Dgms KerDiagrams; //vector of kernel diagrams, one in each dimension poissble (these may be empty)
			int max_dim; //the maximum dimension of a cell
			std::vector<int> L_to_K; //Map cells in L to cells in K.  
			std::vector<bool> InSubcomplex; //track if a cell is in the subcomplex L
			int number_cells_K; //number of cells in K
			int number_cells_L; //number of cells in L
			std::vector<int> order_change; //order_change[i] is the (unsorted) id in K of the ith cell in the filtration.
		

		public: 

			ImKerReduced(Filtration<Int_, Real_, Int_> K_, Filtration<Int_, Real_, Int_> L_,VRUDecomp F_, VRUDecomp G_, VRUDecomp Im_, VRUDecomp Ker_, std::vector<bool> InSubcomplex_, std::vector<int> L_to_K_, std::vector<int> order_change_) : 
				K (K_),
				L (L_),
				F (F_),
				G (G_),
				Im (Im_),
				Ker (Ker_),
				L_to_K (L_to_K_),
				order_change (order_change_),
				InSubcomplex (InSubcomplex_) { 
					number_cells_K = K.boundary_matrix_full().size();
					number_cells_L = L.boundary_matrix_full().size();
					
			}

			void GenereateImDiagrams() {//Generate the kernel diagrams

			}

			void GenerateKerDiagrams() {//Generate the image diagrams
			std::cout << "Starting to extract the kernel diagrams." << std::endl;
				for (int i = 0; i < max_dim; i++){
					ImDiagrams.push_back(Dgm());
				}
				std::vector<bool> open_point (number_cells_K);
				
				//Get the matrices we need to check the conditions
				MatrixData R_f = F.get_R();
				MatrixData D_f = F.get_D();
				MatrixData V_f = F.get_V();
				MatrixData R_g = G.get_R();
				MatrixData D_g = G.get_D();
				MatrixData V_g = G.get_V();
				MatrixData R_ker = Ker.get_R();
				MatrixData R_im = Im.get_R();
				


				std::cout << "Made it into GenerateImDiagrams" << std::endl;
				for (int i = 0; i < number_cells_K; i++) {
					//TODO: should this be a serpate test?
					//Check if a cell gives birth to a class, need to check if it is negative in R_f
					std::cout << "looking at cell " << i << " in the sorted_id ordering. Cell " << i << " has InSubcomplex " << InSubcomplex[i] << std::endl;

					std::vector<int> K_to_L(number_cells_K);

					for (int i = 0; i < number_cells_L; i++) {
						K_to_L[L_to_K[i]] = i;					
					}

					if (!InSubcomplex[i] && !R_f[i].empty()) { //cell needs to be in L, and needs to not be empty in R_f
						
						std::vector<int> quasi_sum (number_cells_K, 0);
						for (int j = 0; j < V_f[i].size(); j++) {
							for (int k = 0; k < D_f[V_f[i][j]].size(); k++) {
								quasi_sum[D_f[V_f[i][j]][k]]++;//get the boundary in K
							}
						}
						bool cycle = true;
						for (int k = 0; k < quasi_sum.size(); k++) {
							if (quasi_sum[k]%2 !=0) { //check if represents a cycle in K
								cycle = false;
								break;
							}
						}
						std::cout << R_im << std::endl;
						if (!cycle && InSubcomplex[R_im[i].back()]) { // check if lowest entry corresponds to a cell in L
							open_point[i] = true;
							std::cout << "cell " << order_change[i] << " gave birth." << std::endl;
						}
						std::cout << "checked if cell " << order_change[i] << " gave birth." << std::endl;
					}
					//Now check if cell kills a class, in which case find the paired cell, need to check if it is positive in R_f
					else if (InSubcomplex[i] && R_g[i].empty()) {
						std::cout << "now check if cell " << order_change[i] << " kills." << std::endl;

						std::vector<int> quasi_sum (number_cells_K, 0);
						for (int j = 0; j < V_f[i].size(); j++) {
							for (int k = 0; k < D_f[V_f[i][j]].size(); k++) {
								quasi_sum[D_f[V_f[i][j]][k]] += 1;//get the boundary in K
							}
						}
						bool cycle = true;
						for (int k = 0; k < quasi_sum.size(); k++) {
							if (quasi_sum[k]%2 !=0) { //check if represents a cycle in K
								cycle = false;
								break;
							}
						}
						if (cycle && !R_g[K_to_L[i]].empty()) { //if it is a cycle, we check for negative in R_g, which requires R_g[i] to be not empty
							//next check if cycle or not in L
							std::vector<int> quasi_sum (number_cells_L, 0);
							for (int j = 0; j < V_g[K_to_L[i]].size(); j++) {
								for (int k = 0; k < D_g[V_g[K_to_L[i]][j]].size(); k++) {
									quasi_sum[D_g[V_g[K_to_L[i]][j]][k]] += 1;//get the boundary in K
								}
							}
							for (int k = 0; k < quasi_sum.size(); k++) {
								if (quasi_sum[k]%2 != 0) {
									int birth_id = R_ker[i].back();
									int dim = K.dim_by_id(i)-1; 
									std::cout << "This represents a point in dimension " << dim << std::endl;
									ImDiagrams[dim].push_back(Point(K.get_simplex_value(order_change[birth_id]), K.get_simplex_value(order_change[i])));
									open_point[birth_id] = false;
									std::cout << "Have something that kills a class, and it is in dimension " << dim << std::endl;
								}
							}
						}
					}
				}

				for (int i = 0; i < open_point.size(); i++) {
					if (open_point[i]) {
						int dim = K.dim_by_sorted_id(i)-1;//(order_change[i])-1;
						std::cout << i << " The cell " << order_change[i] << " is an open point in dimension " << dim << " so we added a point (" << K.value_by_sorted_id(i) <<", " << std::numeric_limits<double>::infinity() << ")." << std::endl;
						ImDiagrams[dim].push_back(Point(K.value_by_sorted_id(i), std::numeric_limits<double>::infinity()));
					}
				}

			}

			

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

			void set_max_dim(int d) {
				max_dim = d;
			}
	};


	

	template <typename Int_, typename Real_>
	ImKerReduced<Int_, Real_> reduce_im_ker(Filtration<Int_, Real_, Int_> K, Filtration<Int_, Real_, Int_> L, std::vector<int> L_to_K, Params& params) {//L_to_K maps a cell in L to a cell in K
		using Real = Real_;
		using Int = Int_;
    	using IntSparseColumn = SparseColumn<Int>;
    	using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
   		using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;
		using Point = DgmPoint<Real>;
		using Diagram = std::vector<Point>;

		/*std::cout << "Need to check the values in K:" << std::endl;
		for (int i = 0; i < K.boundary_matrix_full().size(); i++){
			std::cout << "Cell " << i << " has value " << K.get_simplex_value(i) << std::endl;
		}*/
		VRUDecomp F(K.boundary_matrix_full());
		F.reduce_parallel_rvu(params);
		//std::cout << "F sanity check." << std::endl;
		//F.sanity_check();

		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);
		//std::cout << "G sanity check." << std::endl;
		//G.sanity_check();

		

		FiltrationSimplexVector K_simps = K.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_K =  K_simps.size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_L =  L_simps.size(); // number of simplices in L

		/*std::cout << "Need to check the values in K:" << std::endl;
		for (int i = 0; i < number_cells_K; i++){
			std::cout << "Cell " << i << " has value " << K.get_simplex_value(i) << std::endl;
		}*/

		std::vector<bool> InSubcomplex(number_cells_K, false); //We need to keep track of which cells of K are in L, and we should do this with the sorted ids as it will make life easier later.
		std::vector<int> sorted_id_to_id(number_cells_K);
		for (int i = 0; i < number_cells_K; i++) {
			sorted_id_to_id[K.get_sorted_id(i)] = i;
		}

		std::cout << "The cells are in the following sorted order:";
		for (int i = 0; i < number_cells_K; i++) {
			std::cout << " " << sorted_id_to_id[i] ;
		}
		std::cout << std::endl;
		for (int i = 0; i < L_to_K.size(); i++) {
			InSubcomplex[L_to_K[sorted_id_to_id[i]]] = true;
		}

		std::cout << "InSubcomplex has size " << InSubcomplex.size() << std::endl;
					
		std::vector<int> to_del;
		std::vector<int> new_order (number_cells_K);
		std::iota (new_order.begin(), new_order.end(), 0);


		std::vector<int> MapKtoL(number_cells_L);
		for (int i = 0; i < number_cells_L; i++) {
			MapKtoL[L_to_K[i]] = i;	
		}
	/*
		std::cout << "Current order is: ";
		for (int i = 0; i < new_order.size(); i++) {
			std::cout << new_order[i] << " ";
 		}
		std::cout << std::endl;
		std::cout << "The sort seems to be wrong. Let us have a closer look." << std::endl; 

		std::cout << "Need to understand what happens when the cells are sorted. Let us start with K." << std::endl;
		for (int i = 0; i < K.size(); i++) {
			std::cout << "cell " << i << " has sorted id " << K.get_sorted_id(i) << std::endl;
		}
*/
		std::sort(new_order.begin(), new_order.end(), [&](int i, int j) {
			std::cout << "Comparing " << i << " and " << j << std::endl;
			if (InSubcomplex[i] && InSubcomplex[j]) {//FIXME: this needs to work with the sorted order. 
				std::cout << "Both are in the sub complex, so we need to sort by their dimensions and values in L: ";
				int i_dim, j_dim;
				double i_val, j_val;
				for (int k = 0; k < L_to_K.size(); k++) {
					if (L_to_K[k] == i) {
						i_dim = L.dim_by_sorted_id(k);
						i_val = L.value_by_sorted_id(k);
					}
					if (L_to_K[k] == j) {
						j_dim = L.dim_by_sorted_id(k);
						j_val = L.value_by_sorted_id(k);
					}
				}
				std::cout << i_dim << " and " << j_dim << " vs " << i_val << " and " << j_val << std::endl;
				if (i_dim == j_dim) {
					return i_val < j_val;
				} else {
					return i_dim < j_dim;
				}
			} else if (InSubcomplex[i] && !InSubcomplex[j]) {
				std::cout << i << " is in the subcomplex but " << j << " is not" << std::endl;
				return true;
			} else if (!InSubcomplex[i] && InSubcomplex[j]) {
				std::cout << i <<" is not in the subcomplex but " << j << " is"<< std::endl;
				return false;
			} else {
				std::cout << "Neither are in the sub complex, so sorting by their dimensions and values in K: " << std::endl;
				int i_dim, j_dim;
				double i_val, j_val;
				i_dim = K.dim_by_sorted_id(i);
				i_val = K.value_by_sorted_id(i);
				j_dim = K.dim_by_sorted_id(j);
				j_val = K.value_by_sorted_id(j);
				std::cout << i_dim << " and " << j_dim << " vs " << i_val << " and " << j_val << std::endl;
				if (i_dim == j_dim) {
					return i_val < j_val;
				} else {
					return i_dim < j_dim;
				}
			}
		});

		std::cout << "New order is: ";
		for (int i = 0; i < new_order.size(); i++) {
			std::cout << new_order[i] << " ";
 		}
		std::cout << std::endl;
		std::vector<int> old_to_new_order(number_cells_K);

		for (int i = 0; i < number_cells_K; i++) {
			old_to_new_order[new_order[i]] = i;
		}

		MatrixData D_im;
		for (int i = 0; i < F.d_data.size(); i++) {
			std::vector<int> new_col_i;
			std::cout << "Looking at column " << i << " which is currently [";
			for (int j = 0; j < F.d_data[i].size(); j++) {
				std::cout << " " << F.d_data[i][j];
			}
			std::cout << "], and now it is [";
			if (!F.d_data[i].empty()) {
				for (int j = 0; j < F.d_data[i].size(); j++) {
					new_col_i.push_back(new_order[F.d_data[i][j]]);
					std::cout << " " << new_order[F.d_data[i][j]];
				}
			}
			std::cout << "]" << std::endl;
			D_im.push_back(new_col_i);
		}

		VRUDecomp Im(D_im);
		Im.reduce_parallel_rvu(params);

		
	
		MatrixData V_im = Im.get_V();//FIXME: need to fix these quasi sums
    	MatrixData R_im = Im.get_R();

    	std::cout<< "===============" << std::endl;
    	std::cout << "D_im is:" << std::endl;
		for (int i = 0; i < D_im.size(); i++) {
			std::cout << "[ ";
			for (int j = 0; j < D_im[i].size(); j++) {
				std::cout << D_im[i][j] << " ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << "R_im is:" << std::endl;
		for (int i = 0; i < R_im.size(); i++) {
			std::cout << "[ ";
			for (int j = 0; j < R_im[i].size(); j++) {
				std::cout << R_im[i][j] << " ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << "V_im is:" << std::endl;
		for (int i = 0; i < V_im.size(); i++) {
			std::cout << "[ ";
			for (int j = 0; j < V_im[i].size(); j++) {
				std::cout << V_im[i][j] << " ";
			}
			std::cout << "]" << std::endl;
		}
		for (int i = 0; i < V_im.size(); i++) {
			bool del = false;
			std::vector<int> quasi_sum (number_cells_K, 0);
			if (!V_im[i].empty()) {
				for (int j = 0; j < V_im[i].size(); j++) {
					for (int k = 0; k < D_im[V_im[i][j]].size(); k++) {
						quasi_sum[D_im[V_im[i][j]][k]]++;
					}
				}
			}
			std::cout << "quasi_sum is " ;
			for (int j = 0; j < quasi_sum.size(); j++) {
				std::cout << " " <<quasi_sum[j];
			}
			//std::cout << std::endl;
			for (int j = 0; j < quasi_sum.size(); j++) {
				if (quasi_sum[j]%2 !=0) {
					del = true;
					break;
				}
			}
			std::cout << std::endl;
			to_del.push_back(del);
			
		}

		std::cout << "=================" << std::endl << "to_del is: " << std::endl;
		for (int i = 0; i < to_del.size(); i++) {
			std::cout << to_del[i] << std::endl;
		}

		MatrixData d_ker;

		for (int i = 0; i < to_del.size(); i++) {
			if (!to_del[i]) {
				d_ker.push_back(V_im[i]);
			}

		}

		// We need to keep the matrix square, so we also need to delete some rows. But we need to know which rows correspond to the columns we are deleting.
		std::vector<int> rows_to_del;
		for (int i = 0; i < to_del.size(); i++) {
			if (to_del[i])
			rows_to_del.push_back(old_to_new_order[i]);
		}


		std::sort(rows_to_del.begin(), rows_to_del.end(), std::greater<int>());

		std::cout << "We need to delete the following rows:";
		for (int i = 0; i < rows_to_del.size(); i++) {
			std::cout << " " << rows_to_del[i];
		}
		std::cout << std::endl;
		std::cout<< "===============" << std::endl;
    	std::cout << "d_ker is:" << std::endl;
    	for (int i = 0; i < d_ker.size(); i++) {
        std::cout << "[ ";
        for (int j = 0; j < d_ker[i].size(); j++) {
            std::cout << d_ker[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    	}
		std::cout<< "===============" << std::endl;
		for (int i = 0; i < d_ker.size(); i++) {
			for (int k = 0; k < rows_to_del.size(); k++) {
				std::cout << "Deleting row " << rows_to_del[k] << " and d_ker[" << i<< "] is currently ";
				for (int j = 0; j < d_ker[i].size(); j++) {
            		std::cout << d_ker[i][j] << " ";
        		}
				for (int j = 0; j < d_ker[i].size(); j++) {
					if (d_ker[i][j] > k) {
						d_ker[i][j]--;
					} else if (d_ker[i][j] == k) {
						d_ker[i].erase(d_ker[i].begin()+j);
					}
				}
			}
		}
		std::cout<< "===============" << std::endl;
    	std::cout << "d_ker is:" << std::endl;
    	for (int i = 0; i < d_ker.size(); i++) {
        std::cout << "[ ";
        for (int j = 0; j < d_ker[i].size(); j++) {
            std::cout << d_ker[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    	}
		std::cout<< "===============" << std::endl;

		VRUDecomp Ker(d_ker);
		std::cout << "Constructed Ker." << std::endl;
		std::cout << "I really need to figure out what is going wrong with the parallel reduction functionb." << std::endl;
		Ker.reduce_parallel_rvu(params);
		std::cout << "Ker sanity check." << std::endl;
		Ker.sanity_check();

		ImKerReduced<Int, Real> IKR(K, L, F, G, Im, Ker, InSubcomplex, L_to_K, new_order);

		IKR.GenerateKerDiagrams();
		//IKR.GenereateImDiagrams();

		return  IKR;
	}

	template<typename Int_, typename Real_>
	struct CokReduced {
		using Int = Int_;
		using Real = Real_;
        using IntSparseColumn = SparseColumn<Int>;
        using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
    	using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;
		using Point = DgmPoint<Real>;
    	using Dgm = std::vector<Point>;
		using Dgms = std::vector<Dgm>;

		private:

			VRUDecomp F;
			VRUDecomp G;
			VRUDecomp Cok;
			Dgms CokDiagrams;
			std::vector<bool> InSubcomplex;

		public: 

			CokReduced(VRUDecomp F_, VRUDecomp G_, VRUDecomp Cok_, std::vector<int> L_to_K) : 
				F (F_),
				G (G_),
				Cok (Cok_) { 
				std::vector<bool> InSubcomplex(F.get_D().size(), false);
					for (int i = 0; i < L_to_K.size(); i++) {
						InSubcomplex[L_to_K[i]] = true;
					}
			}

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

			MatrixData get_D_cok() {
				return Cok.get_D();
			}

			MatrixData get_V_cok() {
				return Cok.get_V();
			}

			MatrixData get_R_cok() {
				return Cok.get_R();
			}
	};

	template <typename Int_, typename Real_>
	CokReduced<Int_, Real_> reduce_cok(Filtration<Int_, Real_, Int_> K, Filtration<Int_, Real_, Int_> L, std::vector<int> L_to_K, Params& params) {//FilteredPair<Int_, Real_> KL) {
		using Int = Int_;
		using Real = Real_;
		using Int = Int_;
    	using IntSparseColumn = SparseColumn<Int>;
    	using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
   		using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;

		VRUDecomp F(K.boundary_matrix_full());
		F.reduce_parallel_rvu(params);
		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);

		int number_cells_K =  K.simplices().size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_L =  L_simps.size(); // number of simplices in L
					
		MatrixData D_cok(F.get_D());
		MatrixData D_g(G.get_D());
		MatrixData V_g(G.get_V());
		for (int i = 0; i < V_g.size(); i++) {
			bool replace = true;
			std::vector<int> quasi_sum (number_cells_L, 0);
			if (!V_g[i].empty()) {
				for (int j = 0; j < V_g[i].size(); j++) {
					for (int k = 0; k < D_g[L_to_K[V_g[i][j]]].size(); k++) {
						quasi_sum[D_g[L_to_K[V_g[i][j]]][k]] += 1;//check if a column in V_g represents a cycle
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
				D_cok[L_to_K[i]] = V_g[i]; 
			}
		}

		VRUDecomp Cok(D_cok);
		Cok.reduce_parallel_rvu(params);

		CokReduced<Int, Real> CkR(F, G, Cok, L_to_K);
		return  CkR;
		
	}

}
