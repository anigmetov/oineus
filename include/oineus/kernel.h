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
	struct ImKerCokReduced {
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
			VRUDecomp Cok; //reduced cokernel triple
			Dgms KerDiagrams; //vector of image diagrams, one in each dimension poissble (these may be empty)
			Dgms ImDiagrams; //vector of kernel diagrams, one in each dimension poissble (these may be empty)
			Dgms CokDiagrams; //vector of kernel diagramsm, one in each possible dimension (these may be empty)
			int max_dim; //the maximum dimension of a cell
			std::vector<int> L_to_K;
			std::vector<bool> InSubcomplex; //track if a cell is in the subcomplex L
			int number_cells_K; //number of cells in K
			int number_cells_L; //number of cells in L
			std::vector<int> OrderChange; //OrderChange[i] is the (unsorted) id in K of the ith cell in the filtration.
		

		public: 

			ImKerCokReduced(Filtration<Int_, Real_, Int_> K_, Filtration<Int_, Real_, Int_> L_,VRUDecomp F_, VRUDecomp G_, VRUDecomp Im_, VRUDecomp Ker_, VRUDecomp Cok_, std::vector<bool> InSubcomplex_, std::vector<int> L_to_K_, std::vector<int> OrderChange_) : 
				K (K_),
				L (L_),
				F (F_),
				G (G_),
				Im (Im_),
				Ker (Ker_),
				Cok (Cok_),
				L_to_K (L_to_K_),
				OrderChange (OrderChange_),
				InSubcomplex (InSubcomplex_) { 
					number_cells_K = K.boundary_matrix_full().size();
					number_cells_L = L.boundary_matrix_full().size();
					max_dim = 1;//K.max_dim();
			}

			void GenereateImDiagrams(std::vector<int> new_cols) {//Generate the kernel diagrams
				std::cout << "Starting to extract the image diagrams." << std::endl;

				for (int i = 0; i < max_dim+1; i++){
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
				MatrixData R_im = Im.get_R();

				std::vector<int> sorted_id_to_id(number_cells_K, 0);
				for (int i = 0; i < number_cells_K; i++) {
					sorted_id_to_id[K.get_sorted_id(i)] = i;
				}

				std::vector<int> SubComplex(number_cells_K, -1);

				for (int i = 0; i < number_cells_L; i++) {
					SubComplex[K.get_sorted_id(L_to_K[i])] = i;
				}

				for (int i = 0; i < number_cells_K; i++) {
					int id_in_L = SubComplex[i];
				}

				std::vector<bool> open_points(number_cells_L, false);

				std::cerr << "R_g is " << R_g << std::endl;
				for (int i = 0; i < number_cells_K; i++) {
					int id_in_L = SubComplex[i];
					if (id_in_L != -1 && R_g[i].empty()) {//the cell needs to be in L, and negative in R_g, which requires R_g to be empty, and then we check if the column in V_g stores a cycle.

						std::vector<int> quasi_sum_g(number_cells_L, 0);
						for (int j = 0; j < V_g[i].size(); j++) {
							for (int k = 0; k < D_g[V_g[i][j]].size(); k++) {
								quasi_sum_g[D_g[V_g[i][j]][k]]++;
							}
						}

						std::cerr << "for " << i << " quasi_sum_g is ";
						
						for (int j = 0; j < quasi_sum_g.size(); j++) {
							std::cerr << " " << quasi_sum_g[j];
						}
						std::cerr << std::endl;
						bool cycle_g = true;
						for (int j = 0; j < number_cells_L; j++) {
							if (quasi_sum_g[j] %2 != 0) {
								cycle_g = false;
								break;
							}
						}

						if (cycle_g) {
							open_points[id_in_L] = true;
						}

						std::cerr << "for " << i << " cycle_g is " << cycle_g << std::endl;
					} else if (!R_f[i].empty()) {
						std::vector<int> quasi_sum_f(number_cells_K, 0);
						std::cerr << " looking at " << V_f[i] << std::endl;

						for (int j = 0; j < V_f[i].size(); j++) {
							std::cerr << " looking at " << V_f[i][j] << " and " << D_f[V_f[i][j]]<< std::endl;
							for (int k = 0; k < D_f[V_f[i][j]].size(); k++ ) {
								quasi_sum_f[D_f[V_f[i][j]][k]]++;
								std::cerr << "quasi_sum_f[" << D_f[V_f[i][j]][k] << "] increased to " << quasi_sum_f[D_f[V_f[i][j]][k]] << std::endl;
							}
						}

						bool cycle_f = true;

						for (int j = 0; j < quasi_sum_f.size(); j++) {
							if (quasi_sum_f[j] %2 != 0) {
								cycle_f = false;
								break;
							}
						}
						std::cerr << "for " << i << " cycle_f is " << cycle_f << std::endl;

						if (!cycle_f && SubComplex[OrderChange[R_im[i].back()]] != -1) {
							int birth_id = OrderChange[R_im[i].back()];
							int dim = K.dim_by_id(i)-1; 
							ImDiagrams[dim].push_back(Point(K.value_by_sorted_id(birth_id), K.value_by_sorted_id(i))); //K.value_by_sorted_id(i)
							std::cerr << "Found a cycle which should kill something, it has id " << sorted_id_to_id[i] << " and the thing it kills was born by " << birth_id << " which has open point " << open_point[birth_id] << std::endl;
							open_point[birth_id] = false;
						}
					}
				} 

				std::cerr << "Image open points are ";
				for (int i = 0; i < open_points.size(); i++) {
					std::cerr << " " << open_points[i];
				}
				std::cerr << std::endl;
				for (int i = 0; i < open_points.size(); i++) {
					if (open_points[i]) {
						int dim = K.dim_by_sorted_id(i);//(OrderChange[i])-1;
						std::cerr << i << " is open with dimension " << dim << " and value " << K.value_by_sorted_id(i) << std::endl;
						ImDiagrams[dim].push_back(Point(K.value_by_sorted_id(i), std::numeric_limits<double>::infinity()));
					}
				}
			
				for (int i = 0; i < KerDiagrams.size(); i++){
					if (ImDiagrams[i].empty()) {
					std::cout << "Image diagram in dimension " << i << " is empty." << std::endl;

					} else { 
						std::cout << "Image diagram in dimension " << i << " is: " << std::endl;
						for (int j = 0; j < ImDiagrams[i].size(); j++) {
							std::cout << "(" << ImDiagrams[i][j].birth << ", " << ImDiagrams[i][j].death << ")" << std::endl;
						}
					}
				}

			}

			void GenerateKerDiagrams(std::vector<int> new_cols) {//Generate the image diagrams
				std::cout << "Starting to extract the kernel diagrams." << std::endl;
				
				for (int i = 0; i < max_dim+1; i++){
					KerDiagrams.push_back(Dgm());
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
				
				std::vector<int> sorted_id_to_id(number_cells_K);

				for (int i = 0; i < number_cells_K; i++) {
					sorted_id_to_id[K.get_sorted_id(i)] = i;
				}

				std::vector<int> SubComplex(number_cells_K, -1);

				for (int i = 0; i < number_cells_L; i++) {
					SubComplex[K.get_sorted_id(L_to_K[i])] = i;
				}

				for (int i = 0; i < number_cells_K; i++) {
					//TODO: should this be a serpate test?
					//Check if a cell gives birth to a class, need to check if it is negative in R_f
					int id_in_L = SubComplex[i];
					if (id_in_L == -1 && !R_f[i].empty()) { //cell needs to be in K\L, and needs to not be empty in R_f
						
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
						if (!cycle && SubComplex[R_im[i].back()] != -1) { // check if lowest entry corresponds to a cell in L
							open_point[sorted_id_to_id[i]] = true;
						}
					}
					//Now check if cell kills a class, in which case find the paired cell, need to check if it is positive in R_f

					else if (id_in_L != -1 && !R_g[SubComplex[i]].empty()) {
						std::vector<int> quasi_sum_g (number_cells_L, 0);
						
						for (int j = 0; j < V_g[id_in_L].size(); j++) {
							for (int k = 0; k < D_g[V_g[id_in_L][j]].size(); k++) {
								quasi_sum_g[D_g[V_g[id_in_L][j]][k]]++;//get the boundary in K
							}
						}
						bool cycle = true;
						for (int k = 0; k < quasi_sum_g.size(); k++) {
							if (quasi_sum_g[k]%2 !=0) { //check if represents a cycle in K
								cycle = false;
								break;
							}
						}
						
						if (!cycle && R_f[i].empty()) {
							std::vector<int> quasi_sum_f (number_cells_K, 0);
							for (int j = 0; j < V_f[i].size(); j++) {
								for (int k = 0; k < D_f[V_f[i][j]].size(); k++) {
									quasi_sum_f[D_f[V_f[i][j]][k]]++;//get the boundary in K
								}
							}
							bool cycle_f = true;
							
							for (int k = 0; k < quasi_sum_f.size(); k++) {
								if (quasi_sum_f[k]%2 != 0) {
									cycle_f = false;
									break;									
								}
							}
							
							
							if (cycle_f) {
								
								int birth_id = OrderChange[R_ker[new_cols[i]].back()];
								int dim = K.dim_by_id(i)-1; 
								KerDiagrams[dim].push_back(Point(K.value_by_sorted_id(K.get_sorted_id(birth_id)), K.value_by_sorted_id(i))); //K.value_by_sorted_id(i)
								open_point[birth_id] = false;
								
							}
						}
					}
				}
				

				for (int i = 0; i < open_point.size(); i++) {
					if (open_point[i]) {
						int dim = K.dim_by_sorted_id(i)-1;//(OrderChange[i])-1;
						KerDiagrams[dim].push_back(Point(K.value_by_sorted_id(i), std::numeric_limits<double>::infinity()));
					}
				}
			
				for (int i = 0; i < KerDiagrams.size(); i++){
					if (KerDiagrams[i].empty()) {
					std::cout << "Kernel diagram in dimension " << i << " is empty." << std::endl;

					} else { 
						std::cout << "Kernel diagram in dimension " << i << " is: " << std::endl;
						for (int j = 0; j < KerDiagrams[i].size(); j++) {
							std::cout << "(" << KerDiagrams[i][j].birth << ", " << KerDiagrams[i][j].death << ")" << std::endl;
						}
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
	ImKerCokReduced<Int_, Real_> reduce_im_ker_cok(Filtration<Int_, Real_, Int_> K, Filtration<Int_, Real_, Int_> L, std::vector<int> L_to_K, Params& params) {//L_to_K maps a cell in L to a cell in K
		using Real = Real_;
		using Int = Int_;
    	using IntSparseColumn = SparseColumn<Int>;
    	using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
   		using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;
		using Point = DgmPoint<Real>;
		using Diagram = std::vector<Point>;


		FiltrationSimplexVector K_simps = K.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_K =  K_simps.size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_L =  L_simps.size(); // number of simplices in L

		std::vector<bool> InSubcomplex(number_cells_K, false); //We need to keep track of which cells of K are in L, and we should do this with the sorted ids as it will make life easier later.
		std::vector<int> sorted_id_to_id(number_cells_K);
		for (int i = 0; i < number_cells_K; i++) {
			sorted_id_to_id[K.get_sorted_id(i)] = i;
		}

		for (int i = 0; i < L_to_K.size(); i++) {
			InSubcomplex[L_to_K[sorted_id_to_id[i]]] = true;
		}

		VRUDecomp F(K.boundary_matrix_full());
		F.reduce_parallel_rvu(params);

		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);
	
		std::vector<int> to_del(number_cells_K);
		std::vector<int> new_order (number_cells_K);
		std::iota (new_order.begin(), new_order.end(), 0);


		std::vector<int> MapKtoL(number_cells_K);
		for (int i = 0; i < number_cells_L; i++) {
			MapKtoL[L_to_K[i]] = i;	
		}

		std::sort(new_order.begin(), new_order.end(), [&](int i, int j) {
			if (InSubcomplex[i] && InSubcomplex[j]) {//FIXME: this needs to work with the sorted order. 
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
				if (i_dim == j_dim) {
					return i_val < j_val;
				} else {
					return i_dim < j_dim;
				}
			} else if (InSubcomplex[i] && !InSubcomplex[j]) {
				return true;
			} else if (!InSubcomplex[i] && InSubcomplex[j]) {
				return false;
			} else {
				int i_dim, j_dim;
				double i_val, j_val;
				i_dim = K.dim_by_sorted_id(i);
				i_val = K.value_by_sorted_id(i);
				j_dim = K.dim_by_sorted_id(j);
				j_val = K.value_by_sorted_id(j);
				if (i_dim == j_dim) {
					return i_val < j_val;
				} else {
					return i_dim < j_dim;
				}
			}
		});

		std::vector<int> old_to_new_order(number_cells_K);

		for (int i = 0; i < number_cells_K; i++) {
			old_to_new_order[new_order[i]] = i;
		}

		MatrixData D_im;
		for (int i = 0; i < F.d_data.size(); i++) {
			std::vector<int> new_col_i;
			if (!F.d_data[i].empty()) {
				for (int j = 0; j < F.d_data[i].size(); j++) {
					new_col_i.push_back(new_order[F.d_data[i][j]]);
				}
			}
			D_im.push_back(new_col_i);
		}

		VRUDecomp Im(D_im);
		Im.reduce_parallel_rvu(params);
	
		MatrixData V_im = Im.get_V();
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
		
			for (int j = 0; j < quasi_sum.size(); j++) {
				if (quasi_sum[j]%2 !=0) {
					del = true;
					break;
				}
			}
			to_del[i] = del;
			
		}

		MatrixData d_ker;

		for (int i = 0; i < to_del.size(); i++) {
			if (!to_del[i]) {
				d_ker.push_back(V_im[i]);
			}

		}

		std::vector<int> new_cols(number_cells_K, -1);
		int counter = 0;

		for (int i = 0; i < number_cells_K; i++) {
			if (to_del[i] != 1) {
				new_cols[i] = counter;
				counter++;
			}
		}

		
		VRUDecomp Ker(d_ker, K.size());
		Ker.reduce_parallel_rvu(params);

		MatrixData D_cok(F.get_D());
		MatrixData D_g = G.get_D();
		MatrixData R_g = G.get_R();
		MatrixData V_g = G.get_V();

		for (int i = 0; i < V_g.size(); i++) {
			bool replace = true;
			std::vector<int> quasi_sum (number_cells_L, 0);
			if (!V_g[i].empty()) {
				for (int j = 0; j < V_g[i].size(); j++) {
					for (int k = 0; k < D_g[L_to_K[V_g[i][j]++]].size(); k++) {
						quasi_sum[D_g[L_to_K[V_g[i][j]]][k]];//check if a column in V_g represents a cycle
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

		ImKerCokReduced<Int, Real> IKCR(K, L, F, G, Im, Ker, Cok, InSubcomplex, L_to_K, new_order);	

		IKCR.GenerateKerDiagrams(new_cols);
		IKCR.GenereateImDiagrams(new_cols);

		return  IKCR;
	}

	/*template<typename Int_, typename Real_>
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

			Filtration<Int_, Real_, Int_> K;
			Filtration<Int_, Real_, Int_> L;
			VRUDecomp F;
			VRUDecomp G;
			VRUDecomp Cok;
			Dgms CokDiagrams;
			int max_dim; //the maximum dimension of a cell
			std::vector<int> L_to_K;
			std::vector<bool> InSubcomplex; //track if a cell is in the subcomplex L
			int number_cells_K; //number of cells in K
			int number_cells_L; //number of cells in L
		

		public: 

			CokReduced(Filtration<Int_, Real_, Int_> K_, Filtration<Int_, Real_, Int_> L_,VRUDecomp F_, VRUDecomp G_, VRUDecomp Cok_, std::vector<bool> InSubcomplex_, std::vector<int> L_to_K_) : 
				K (K_),
				L (L_),
				F (F_),
				G (G_),
				Cok (Cok_),
				InSubcomplex (InSubcomplex_) { 
					number_cells_K = K.boundary_matrix_full().size();
					number_cells_L = L.boundary_matrix_full().size();
					max_dim = 1;//K.max_dim();
			}

			void GenerateCokDiagrams() {
				std::cerr << "Starting to extract cokernel images." << std::endl;

				for (int i = 0; i < max_dim; i++) {
					CokDiagrams.push_back(Dgm ());
				}

				std::cerr << "Prepared " << max_dim << " diagrams." << std::endl;

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

		

		FiltrationSimplexVector K_simps = K.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_K =  K_simps.size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int number_cells_L =  L_simps.size(); // number of simplices in L

		std::vector<bool> InSubcomplex(number_cells_K, false); //We need to keep track of which cells of K are in L, and we should do this with the sorted ids as it will make life easier later.
		std::vector<int> sorted_id_to_id(number_cells_K);
		for (int i = 0; i < number_cells_K; i++) {
			sorted_id_to_id[K.get_sorted_id(i)] = i;
		}

		for (int i = 0; i < L_to_K.size(); i++) {
			InSubcomplex[L_to_K[sorted_id_to_id[i]]] = true;
		}

		VRUDecomp F(K.boundary_matrix_full());
		F.reduce_parallel_rvu(params);

		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);

					
		MatrixData D_cok(F.get_D());
		MatrixData D_g(G.get_D());
		MatrixData V_g(G.get_V());
		for (int i = 0; i < V_g.size(); i++) {
			bool replace = true;
			std::vector<int> quasi_sum (number_cells_L, 0);
			if (!V_g[i].empty()) {
				for (int j = 0; j < V_g[i].size(); j++) {
					for (int k = 0; k < D_g[L_to_K[V_g[i][j]++]].size(); k++) {
						quasi_sum[D_g[L_to_K[V_g[i][j]]][k]];//check if a column in V_g represents a cycle
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

		CokReduced<Int, Real> CkR(K, L, F, G, Cok, InSubcomplex, L_to_K);
		
		CkR.GenerateCokDiagrams();

		return  CkR;
		
	}
	*/
}

