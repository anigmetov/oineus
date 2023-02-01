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

			VRUDecomp F;
			VRUDecomp G;
			VRUDecomp Im;
			VRUDecomp Ker;
			Dgms ImDiagrams;
			Dgms KerDiagrams;
			int max_dim;
			std::vector<int> L_K_sorted_to_id;
		

		public: 

			ImKerReduced(VRUDecomp F_, VRUDecomp G_, VRUDecomp Im_, VRUDecomp Ker_) : 
				F (F_),
				G (G_),
				Im (Im_),
				Ker (Ker_) { 
		
			}

			void GenerateImDiagrams() {//Generate the image diagrams
				 std::vector<Dgms> ImDiagrams (max_dim);
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
	ImKerReduced<Int_, Real_> reduce_im_ker(Filtration<Int_, Real_, Int_> K, Filtration<Int_, Real_, Int_> L, std::vector<int> IdMapping, Params& params) {//FilteredPair<Int_, Real_> KL) {
		using Int = Int_;
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
		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);

		int n_simps_K =  K.simplices().size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int n_simps_L =  L_simps.size(); // number of simplices in L
					
		std::vector<int> to_del;
		std::vector<int> new_order (n_simps_K);
		std::iota (new_order.begin(), new_order.end(), 0);

		for (int i = 0; i < n_simps_L; i++) {
			new_order[IdMapping[i]] = L_simps[i].get_sorted_id();
		}

		MatrixData D_im(G.d_data);

		VRUDecomp Im(D_im);
		Im.reduce_parallel_rvu(params);

		new_order.clear();
		MatrixData V_im = Im.get_V();
		for (int i = 0; i < V_im[0].size(); i++) {
			bool del = true;
			std::vector<int> quasi_sum (n_simps_K, 0);
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
		new_order.clear(); //We have already got everything in the correct order, so have an empty new order to not change anything
		MatrixData D_ker(V_im);//
		VRUDecomp Ker(D_ker);
		Ker.reduce_parallel_rvu(params);

		ImKerReduced<Int, Real> IKR(F, G, Im, Ker);//, IdMapping);

		IKR.GenerateImDiagrams();
		IKR.GenereateKerDiagrams();

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

		public: 

			CokReduced(VRUDecomp F_, VRUDecomp G_, VRUDecomp Cok_) : 
				F (F_),
				G (G_),
				Cok (Cok_)
			{ }

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

		int n_simps_K =  K.simplices().size(); // number of simplices in K
		FiltrationSimplexVector L_simps = L.simplices(); //simplices of L as we will need to work with them to get their order
		int n_simps_L =  L_simps.size(); // number of simplices in L
					
		MatrixData D_cok(F.get_D());
		MatrixData D_g(G.get_D());
		MatrixData V_g(G.get_V());
		for (int i = 0; i < V_g.size(); i++) {
			bool replace = true;
			std::vector<int> quasi_sum (n_simps_L, 0);
			if (!V_g[i].empty()) {
				for (int j = 0; j < V_g[i].size(); j++) {
					for (int k = 0; k < D_g[IdMapping[V_g[i][j]]].size(); k++) {
						quasi_sum[D_g[IdMapping[V_g[i][j]]][k]] += 1;//check if a column in V_g represents a cycle
					}
				}
			}
			for (int j = 0; j < quasi_sum.size(); j++) {
				if (quasi_sum[i]%2 !=0) {
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

		CokReduced<Int, Real> CkR(F, G, Cok);
		return  CkR;
		
	}

}
