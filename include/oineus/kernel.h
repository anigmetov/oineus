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
            std::vector<int> IdMapping;

            FilteredPair(const Filtration<Int, Real, int> K_, const Filtration<Int, Real, int> L_, std::vector<int> IdMapping_, const Params params_) { //If the ids of simplices in L_ do not match their ids in K_ we need to know what the correspondence is.
                K = K_;
                L = L_;
                IdMapping = IdMapping_;
                params = params_;
            }

			FilteredPair(const Filtration<Int_, Real_, int> K_, const Filtration<Int_, Real_, int> L_, const Params params_) { // If the ids of simplices in L_ agree with the ids of simplices in K_ we don't need an IdMapping as it it just the identity
                K = K_;
                L = L_;
                IdMapping = std::vector<int> (L.simplices().size());
				std::iota (IdMapping.begin(), IdMapping.end(), 0);
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

			VRUDecomp F; //the reduced triple for F0
			VRUDecomp G; //reduced triple for G
			VRUDecomp Im; //reduced image triple
			VRUDecomp Ker; //reduced kernel triple
			Dgms ImDiagrams; //vector of image diagrams, one in each dimension poissble (these may be empty)
			Dgms KerDiagrams; //vector of kernel diagrams, one in each dimension poissble (these may be empty)
			int max_dim; //the maximum dimension of a cell
			std::vector<bool> InSubcomplex; //track if a cell is in the subcomplex L
			int number_cells_K; //number of cells in K
			int number_cells_L;
			std::vector<int> OrderChange;
		

		public: 

			ImKerReduced(VRUDecomp F_, VRUDecomp G_, VRUDecomp Im_, VRUDecomp Ker_, std::vector<bool> InSubcomplex_, std::vector<int> IdMapping, std::vector<int> OrderChange_) : 
				F (F_),
				G (G_),
				Im (Im_),
				Ker (Ker_),
				OrderChange (OrderChange_),
				InSubcomplex (InSubcomplex_) { 
					number_cells_K = F_.get_D().size();
					number_cells_L = G_.get_D().size();
					/*std::vector<bool> InSubcomplex(number_cells_K, false);
					for (int i = 0; i < IdMapping.size(); i++) {
						InSubcomplex[IdMapping[i]] = true;
					}*/
			}

			void GenerateImDiagrams() {//Generate the image diagrams
				std::vector<Dgms> ImDiagrams (max_dim);
				std::vector<bool> open_point (number_cells_K);
				
				//Get the matrices we need to check the conditions
				MatrixData R_f = F.get_R();
				MatrixData D_f = F.get_D();
				MatrixData V_f = F.get_V();
				MatrixData R_g = G.get_R();
				MatrixData D_g = G.get_D();
				MatrixData V_g = G.get_V();
				MatrixData R_ker = Ker.get_R();


				std::cout << "Made it into GenerateImDiagrams" << std::endl;
				for (int i = 0; i < number_cells_K; i++) {
					//TODO: should this be a serpate test?
					//Check if a cell gives birth to a class, need to check if it is negative in R_f
					if (!InSubcomplex[i] && !R_f[i].empty()) { //cell needs to be in L, and needs to not be empty in R_f
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
						if (!cycle && R_f[i].back() < number_cells_L) { // check if lowest entry corresponds to a cell in L
							open_point[i] = true;
						}
						std::cout << "checked if cell " << i << " gave birth." << std::endl;
					}
					//Now check if cell kills a class, in which case find the paired cell, need to check if it is positive in R_f
					else if (InSubcomplex[i] && R_g[i].empty()) {
						std::cout << "now check if cell " << i << " kills." << std::endl;

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
						if (cycle && !R_g[i].empty()) { //if it is a cycle, we check for negative in R_g, which requires R_g[i] to be not empty
							//next check if cycle or not in L
							std::vector<int> quasi_sum (number_cells_L, 0);
							for (int j = 0; j < V_g[i].size(); j++) {
								for (int k = 0; k < D_g[V_g[i][j]].size(); k++) {
									quasi_sum[D_g[V_g[i][j]][k]] += 1;//get the boundary in K
								}
							}
							for (int k = 0; k < quasi_sum.size(); k++) {
								if (quasi_sum[k]%2 != 0) {
									//int birth_id;
								}
							}
						}
					}
				}

			}

			void GenereateKerDiagrams() {//Generate the kernel diagrams

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
	ImKerReduced<Int_, Real_> reduce_im_ker(Filtration<Int_, Real_, Int_> K, Filtration<Int_, Real_, Int_> L, std::vector<int> IdMapping, Params& params) {//IdMapping maps a cell in L to a cell in K
		using Real = Real_;
		using Int = Int_;
    	using IntSparseColumn = SparseColumn<Int>;
    	using MatrixData = std::vector<IntSparseColumn>;
		using FiltrationSimplex = Simplex<Int, Real, int>;
   		using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
		using VRUDecomp = VRUDecomposition<Int>;
		using Point = DgmPoint<Real>;
		using Diagram = std::vector<Point>;

		VRUDecomp F(K.boundary_matrix_full());
		F.reduce_parallel_rvu(params);
		std::cout << "F sanity check." << std::endl;
		F.sanity_check();

		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);
		std::cout << "G sanity check." << std::endl;
		G.sanity_check();

		FiltrationSimplexVector K_simps = K.simplices(); //simplices of L as we will need to work with them to get their order
		int n_cells_K =  K_simps.size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int n_cells_L =  L_simps.size(); // number of simplices in L

		std::vector<bool> InSubcomplex(n_cells_K, false); //We need to keep track of which cells of K are in L
		for (int i = 0; i < IdMapping.size(); i++) {
			InSubcomplex[IdMapping[i]] = true;
		}

		std::cout << "InSubcomplex has size " << InSubcomplex.size() << std::endl;
					
		std::vector<int> to_del;
		std::vector<int> NewOrder (n_cells_K);
		std::iota (NewOrder.begin(), NewOrder.end(), 0);


		std::vector<int> MapKtoL(n_cells_L);
		for (int i = 0; i < n_cells_L; i++) {
			MapKtoL[IdMapping[i]] = i;	
		}

		std::cout << "Current order is: ";
		for (int i = 0; i < NewOrder.size(); i++) {
			std::cout << NewOrder[i] << " ";
 		}
		std::cout << std::endl;

		std::sort(NewOrder.begin(), NewOrder.end(), [&](int i, int j) {
			if (InSubcomplex[i] && InSubcomplex[j]) {
				return L.get_simplex_value(i) < L.get_simplex_value(j);
			} else if (InSubcomplex[i] && !InSubcomplex[j]) {
				return true;
			} else if (!InSubcomplex[i] && InSubcomplex[j]) {
				return false;
			} else {
				return K.get_simplex_value(i) < K.get_simplex_value(j);
			}
		});

		std::cout << "New order is: ";
		for (int i = 0; i < NewOrder.size(); i++) {
			std::cout << NewOrder[i] << " ";
 		}
		std::cout << std::endl;
	

		MatrixData D_im(G.d_data);

		VRUDecomp Im(D_im);
		Im.reduce_parallel_rvu(params);
		std::cout << "Im sanity check." << std::endl;
		Im.sanity_check();

		//NewOrder.clear();
		MatrixData V_im = Im.get_V();
		for (int i = 0; i < V_im[0].size(); i++) {
			bool del = true;
			std::vector<int> quasi_sum (n_cells_K, 0);
			if (!V_im[i].empty()) {
				for (int j = 0; j < V_im[i].size(); j++) {
					for (int k = 0; k < D_im[V_im[i][j]].size(); k++) {
						quasi_sum[D_im[V_im[i][j]][k]] += 1;
					}
				}
			}
			for (int j = 0; j < quasi_sum.size(); j++) {
				if (quasi_sum[i]%2 !=0) {
					del = false;
					break;
				}
			}
			if (del) {
				to_del.push_back(i);
			}
		}

		//NewOrder.clear(); //We have already got everything in the correct order, so have an empty new order to not change anything

		MatrixData D_ker(V_im);//
		VRUDecomp Ker(D_ker);
		Ker.reduce_parallel_rvu(params);
		std::cout << "Ker sanity check." << std::endl;
		Ker.sanity_check();

		ImKerReduced<Int, Real> IKR(F, G, Im, Ker, InSubcomplex, IdMapping, NewOrder);

		IKR.GenerateImDiagrams();
		//IKR.GenereateKerDiagrams();

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

			CokReduced(VRUDecomp F_, VRUDecomp G_, VRUDecomp Cok_, std::vector<int> IdMapping) : 
				F (F_),
				G (G_),
				Cok (Cok_) { 
				std::vector<bool> InSubcomplex(F.get_D().size(), false);
					for (int i = 0; i < IdMapping.size(); i++) {
						InSubcomplex[IdMapping[i]] = true;
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
	CokReduced<Int_, Real_> reduce_cok(Filtration<Int_, Real_, Int_> K, Filtration<Int_, Real_, Int_> L, std::vector<int> IdMapping, Params& params) {//FilteredPair<Int_, Real_> KL) {
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

		int n_cells_K =  K.simplices().size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int n_cells_L =  L_simps.size(); // number of simplices in L
					
		MatrixData D_cok(F.get_D());
		MatrixData D_g(G.get_D());
		MatrixData V_g(G.get_V());
		for (int i = 0; i < V_g.size(); i++) {
			bool replace = true;
			std::vector<int> quasi_sum (n_cells_L, 0);
			if (!V_g[i].empty()) {
				for (int j = 0; j < V_g[i].size(); j++) {
					for (int k = 0; k < D_g[IdMapping[V_g[i][j]]].size(); k++) {
						quasi_sum[D_g[IdMapping[V_g[i][j]]][k]] += 1;//check if a column in V_g represents a cycle
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
				D_cok[IdMapping[i]] = V_g[i]; 
			}
		}

		VRUDecomp Cok(D_cok);
		Cok.reduce_parallel_rvu(params);

		CokReduced<Int, Real> CkR(F, G, Cok, IdMapping);
		return  CkR;
		
	}

}
