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
			std::vector<int> sorted_K_to_sorted_L;
			std::vector<int> sorted_L_to_sorted_K;
			//std::vector<bool> InSubcomplex; //track if a cell is in the subcomplex L
			int number_cells_K; //number of cells in K
			int number_cells_L; //number of cells in L
			std::vector<int> new_order_to_old; //new_order_to_old[i] is the (unsorted) id in K of the ith cell in the filtration.
		

		public: 

			ImKerCokReduced(Filtration<Int_, Real_, Int_> K_, Filtration<Int_, Real_, Int_> L_,VRUDecomp F_, VRUDecomp G_, VRUDecomp Im_, VRUDecomp Ker_, VRUDecomp Cok_, std::vector<int> sorted_L_to_sorted_K_, std::vector<int> sorted_K_to_sorted_L_, std::vector<int> new_order_to_old_) : 
				K (K_),
				L (L_),
				F (F_),
				G (G_),
				Im (Im_),
				Ker (Ker_),
				Cok (Cok_),
				sorted_L_to_sorted_K (sorted_L_to_sorted_K_),
				sorted_K_to_sorted_L (sorted_K_to_sorted_L_),
				new_order_to_old (new_order_to_old_) {
					number_cells_K = K.boundary_matrix_full().size();
					number_cells_L = L.boundary_matrix_full().size();
					max_dim = K.max_dim();
					}
			
			void GenerateKerDiagrams(std::vector<int> new_cols) {//Generate the image diagrams
				std::cout << "Starting to extract the kernel diagrams." << std::endl;
				
				for (int i = 0; i < max_dim+1; i++){
					KerDiagrams.push_back(Dgm());
				}

				std::vector<bool> open_points_ker (number_cells_K);

				for (int i = 0; i < number_cells_K; i++) {
                    //std::cerr << "Looking at " << i << " which has sorted_K_to_sorted_L " << sorted_K_to_sorted_L[i] << " and F.get_R()[" << i << "] is " << F.get_R()[i] << std::endl; 
                    if (sorted_K_to_sorted_L[i] == -1) {//to give birth in kernel case, i needs to be in K-L
                        //Next week check if it is negative in F.get_R()
                        if (!F.get_R()[i].empty()) {
                            bool cycle_f = true;
                            //next week check if it represents a cycle
                            std::vector<int> quasi_sum_f(number_cells_K, 0);
                            for (int j = 0; j < F.get_V()[i].size(); j++){
                                for (int k = 0; k < F.get_D()[F.get_V()[i][j]].size(); k++) {
                                    quasi_sum_f[F.get_D()[F.get_V()[i][j]][k]]++;
                                }
                            }

                            for (int j = 0; j < quasi_sum_f.size(); j++) {
                                if (quasi_sum_f[j] %2 != 0) {
                                    cycle_f = false;
                                    break;
                                }
                            }
                            //std::cerr << "cycle_f is " << cycle_f << std::endl;
                            if (!cycle_f) {
                                //std::cerr << "F.get_R()[" << i << "] is " << F.get_R()[i] << " and Im.get_R()[" << i << "] is " << Im.get_R()[i] << std::endl;
                                if (sorted_K_to_sorted_L[F.get_R()[i].back()] != -1) {
                                    //std::cerr << i << " is a cell which gives birth" << std::endl;
                                    open_points_ker[i] = true;
                                }
                            }
                            //std::cerr << "open_points_im is [";
                            //for (int k = 0; k < open_points_ker.size(); k++) {
                            //  std::cerr << " " << open_points_ker[k];
                            //}
                            //std::cerr << "]" << std::endl;
                        }
                    } else if (F.get_R()[i].empty()) { //to kill something, i needs to be positive in f and negative in g, and so we begin with testing positive in f
                        bool cycle_f = true;
                        if (!F.get_V()[i].empty()) {  
                            cycle_f = true;
                            //next week check if it represents a cycle
                            std::vector<int> quasi_sum_f(number_cells_K, 0);
                            for (int j = 0; j < F.get_V()[i].size(); j++){
                                for (int k = 0; k < F.get_D()[F.get_V()[i][j]].size(); k++) {
                                    quasi_sum_f[F.get_D()[F.get_V()[i][j]][k]]++;
                                }
                            }
                            for (int j = 0; j < quasi_sum_f.size(); j++) {
                                if (quasi_sum_f[j] %2 != 0) {
                                    cycle_f = false;
                                    break;
                                }
                            }   
                        }                       
                        //std::cerr << "cycle_f is " << cycle_f << std::endl;
                        if (cycle_f) {
                            //next we check negative in g
                            //std::cerr << "G.get_R() is " << G.get_R() << " and sorted_K_to_sorted_L[" << i << "] is " << sorted_K_to_sorted_L[i]<< std::endl;
                            if (!G.get_R()[sorted_K_to_sorted_L[i]].empty()) {
                                bool cycle_g = true;
                                //next week check if it represents a cycle
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
                                //std::cerr << "cycle_g is " << cycle_g << " and new_cols[" << i << "] is " << new_cols[i] << std::endl;
                                if (!cycle_g) {
                                    //std::cerr << "R_ker is " << R_ker << std::endl;
                                    int birth_id = new_order_to_old[Ker.get_R()[new_cols[i]].back()];
                                    int dim = K.dim_by_sorted_id(i)-1;
                                    if (K.value_by_sorted_id(birth_id) != K.value_by_sorted_id(i)) {
                                        std::cerr << "birth_id is " << birth_id << " and death is " << i << " Found something which kills a class, and that class was born by " << birth_id << " so we add (" << K.value_by_sorted_id(birth_id) << ", " << K.value_by_sorted_id(i) << ") to the dimension " << dim << " diagram" << std::endl;
                                        KerDiagrams[dim].push_back(Point(K.value_by_sorted_id(birth_id), K.value_by_sorted_id(i)));
                                        //std::cerr << "added point " << std::endl;
                                        open_points_ker[birth_id] = false;
                                    }
                                }
                            }
                        }
                    }
                }


				for (int i = 0; i < open_points_ker.size(); i++) {
					if (open_points_ker[i]) {
						int dim = K.dim_by_sorted_id(i)-1;
						//std::cerr << i << " open point with dim " << dim << std::endl;
						KerDiagrams[dim].push_back(Point(K.value_by_sorted_id(i), std::numeric_limits<double>::infinity()));
					}
				}

				std::cerr << "The kernel diagrams are: " << std::endl;
				for (int i = 0; i < KerDiagrams.size(); i++) {
					if (KerDiagrams[i].empty()) {
						std::cerr << "Diagram in dimension " << i << " is empty." << std::endl;
					} else {
						std::cerr << "Diagram in dimension " << i << " is:" << std::endl;
						for (int j = 0; j <KerDiagrams[i].size(); j++) {
							std::cerr << "(" << KerDiagrams[i][j].birth << ", " << KerDiagrams[i][j].death << ")" << std::endl;
						}
					}
				}
				std::cerr << "Finished ker diagrams." << std::endl;
			}

			void GenerateImDiagrams(std::vector<int> new_cols) {//Generate the kernel diagrams
				std::cout << "Starting to extract the image diagrams." << std::endl;

				for (int i = 0; i < max_dim+1; i++){
					ImDiagrams.push_back(Dgm());
				}

				std::vector<bool> open_points_im (number_cells_K);

				for (int i = 0; i < number_cells_K; i++){ 
                    //std::cerr << "looking at " << i;
                    if (sorted_K_to_sorted_L[i] != -1) {//a cell only gives birth if it is in L

                        int sorted_id_in_L = sorted_K_to_sorted_L[i];
                        //std::cerr <<" which has sorted_id_in_L " << sorted_id_in_L << std::endl;
                         //std::cerr << " and G.get_R() is " << G.get_R() << std::endl;
                        //std::cerr << " and G.get_D() is " << G.get_D() << std::endl;
                        //std::cerr << " and G.get_V() is " << G.get_V() << std::endl;


                        if (G.get_R()[sorted_id_in_L].empty()) {//needs to be positive in g
                            std::vector<int> quasi_sum_g(number_cells_L, 0);
                            bool cycle_g = false;
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
                                //std::cerr << "Maybe we should also check if it is positive in F.get_R()?" << std::endl;
                                //std::cerr << "F.get_R() is " << F.get_R() << std::endl;
                                //std::cerr << "F.get_D() is " << F.get_D() << std::endl;
                                //std::cerr << "F.get_V() is " << F.get_V() << std::endl;

                                if (F.get_R()[i].empty()) {//F.get_R()[i] needs to be non-zero
                                    bool cycle_f = false;
                                    if (!F.get_V()[i].empty()) {
                                        cycle_f = true;
                                        std::vector<int> quasi_sum_f(number_cells_K, 0);
                                        for (int j = 0; j < F.get_V()[i].size(); j++) {
                                            for (int k = 0; k < F.get_D()[F.get_V()[i][j]].size(); k++){
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
                                        //std::cerr << i << " is positive in f " << std::endl;
                                    }
                                }

                                //std::cerr << i << " gives birth" << std::endl;
                                //if (Cok.get_R()[i].empty()){
                                    open_points_im[i] = true;
                                //}                   
							}
                        }   
                    }
                    //cells in L and K-L can kill something, so we just test negativitity in 
                    if (!F.get_R()[i].empty()) {//F.get_R()[i] needs to be non-zero
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
                            //std::cerr << i << " is negative in f so we check what the lowest 1 corresponds to." << std::endl;
                            int birth_id = F.get_R()[i].back();
                            if (sorted_K_to_sorted_L[i] != -1) {
                                //std::cerr << birth_id << " is in L." << std::endl;
                                int dim = K.dim_by_sorted_id(i)-1;
                                if (K.value_by_sorted_id(birth_id) != K.value_by_sorted_id(i)) {
                                    ImDiagrams[dim].push_back(Point(K.value_by_sorted_id(birth_id), K.value_by_sorted_id(i)));
                                    //std::cerr << i << " Added the point (" << K.value_by_sorted_id(birth_id) << ", " << K.value_by_sorted_id(i) << ") to dimension " << dim << " diagram." << std::endl;
                                }
                                open_points_im[birth_id] = false;
                            }
                        }
                    }
                }


				//std::cerr << "open_points_im is [";
				//for (int i = 0; i < open_points_im.size(); i++) {
				//	std::cerr << " " << open_points_im[i];
				//}
				//std::cerr << "]" << std::endl;

				for (int i = 0; i < open_points_im.size(); i++) {
					if (open_points_im[i]) {
						int dim = K.dim_by_sorted_id(i);
						ImDiagrams[dim].push_back(Point(K.value_by_sorted_id(i), std::numeric_limits<double>::infinity()));
						std::cerr << i << " Added the point (" << K.value_by_sorted_id(i) << ", " << std::numeric_limits<double>::infinity() << ") to dimension " << dim << " diagram." << std::endl;
					}
				}

				std::cerr << "The image diagrams are: " << std::endl;
				for (int i = 0; i < ImDiagrams.size(); i++) {
					std::cerr << "looking at diagram " << i << std::endl;
					if (ImDiagrams[i].empty()) {
						std::cerr << "Diagram in dimension " << i << " is empty." << std::endl;
					} else if (!ImDiagrams[i].empty()) {
						std::cerr << "Diagram in dimension " << i << " is:" << std::endl;
						for (int j = 0; j < ImDiagrams[i].size(); j++) {
							std::cerr << "(" << ImDiagrams[i][j].birth << ", " << ImDiagrams[i][j].death << ")" << std::endl;
						}
					}
				}

				std::cerr << "Finished image diagrams." << std::endl;
			}
			
			void GenerateCokDiagrams(std::vector<int> new_cols) {//Generate the kernel diagrams
				std::cout << "Starting to extract the cokernel diagrams." << std::endl;

				for (int i = 0; i < max_dim+1; i++){
					CokDiagrams.push_back(Dgm());
				}

				std::vector<bool> open_points_cok (number_cells_K);
				
				//Get the matrices we need to check the conditions

				for (int i = 0; i < number_cells_K; i++) {//we first check that i is positive in f
					std::cerr << "looking at " << i << std::endl;
					if (F.get_R()[i].empty()) {//the column in R needs to be empty
						bool cycle_f = false;
						if (!F.get_V()[i].empty()) {
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
							std::cerr << i << " is positive in f" << std::endl; 
							//now we check if either i is in K-L or negative in g
							if (sorted_K_to_sorted_L[i] == -1) {
								std::cerr << i << " is in K-L and so gives birth" << std::endl;
								open_points_cok[i] = true;
							} else {
								//int id_in_L = sorted_K_to_sorted_L[i];
								//std::cerr << i << " sorted_K_to_sorted_L " << sorted_K_to_sorted_L[i] << " and R_g is " << R_g << " and V_g is " << V_g << " and D_g " << D_g <<std::endl; 
								if (!G.get_R()[sorted_K_to_sorted_L[i]].empty()) {
									bool cycle_g = false;
									if (!G.get_V()[sorted_K_to_sorted_L[i]].empty()) {
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
										std::cerr << sorted_K_to_sorted_L[i] << " is negative in g" << std::endl; 
										open_points_cok[i] = true;
									}
								}
							}
						}
						std::cerr << "finished with " << i << std::endl;
					} else {
						bool cycle_f = false;
						if (!F.get_V()[i].empty()) {
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
						if (!cycle_f) {
							std::cerr << " F.get_R()[" << i << "] is " << F.get_R()[i] << " and Cok.get_R()[i] " << Cok.get_R()[i] << std::endl;
							if (/*sorted_K_to_sorted_L[F.get_R()[i].back()] == -1 &&*/ !Cok.get_R()[i].empty()) {
								int birth_id = Cok.get_R()[i].back();
								//std::cerr << "Found a cell which kills something and that thing was born by " << birth_id << std::endl;
								if (K.value_by_sorted_id(birth_id) != K.value_by_sorted_id(i)) {
									int dim = K.dim_by_sorted_id(birth_id);
									//std::cerr << i << "adding point (" << K.value_by_sorted_id(birth_id) << ", " << K.value_by_sorted_id(i) << ") to the dimension " << dim << " diagram." << std::endl;
								}
								open_points_cok[birth_id] = false;
							}
						}
					}


				}

				for (int i = 0; i < open_points_cok.size(); i++) {
					if (open_points_cok[i]) {
						int dim = K.dim_by_sorted_id(i);
						CokDiagrams[dim].push_back(Point(K.value_by_sorted_id(i), std::numeric_limits<double>::infinity()));
						std::cerr << i << " Added the point (" << K.value_by_sorted_id(i) << ", " << std::numeric_limits<double>::infinity() << ") to dimension " << dim << " diagram." << std::endl;
					}
				}

				std::cerr << "The cokernel diagrams are: " << std::endl;
				for (int i = 0; i < CokDiagrams.size(); i++) {
					if (CokDiagrams[i].empty()) {
						std::cerr << "Diagram in dimension " << i << " is empty." << std::endl;
					} else if (!CokDiagrams[i].empty()) {
						std::cerr << "Diagram in dimension " << i << " is:" << std::endl;
						for (int j = 0; j < CokDiagrams[i].size(); j++) {
							std::cerr << "(" << CokDiagrams[i][j].birth << ", " << CokDiagrams[i][j].death << ")" << std::endl;
						}
					}
				}

				std::cerr << "Finished cokernel diagrams." << std::endl;
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

		std::vector<int> sorted_L_to_sorted_K(number_cells_L, 0);
		std::vector<int> sorted_K_to_sorted_L(number_cells_K, -1);


		for (int i = 0; i < number_cells_L; i++) {
			sorted_L_to_sorted_K[L.get_sorted_id(i)] = K.get_sorted_id(L_to_K[i]);
		}

		std::cerr << "sorted_L_to_sorted_K is [";
		for (int i = 0; i < number_cells_L; i++) {
			std::cerr << " " << sorted_L_to_sorted_K[i];
		}
		std::cerr << "]" << std::endl;

		for (int i = 0; i < number_cells_L; i++) {
			sorted_K_to_sorted_L[sorted_L_to_sorted_K[i]] = i;
		}

		std::cerr << "sorted_K_to_sorted_L is [";
		for (int i = 0; i < number_cells_K; i++) {
			std::cerr << " " << sorted_K_to_sorted_L[i];
		}
		std::cerr << "]" << std::endl;


		VRUDecomp F(K.boundary_matrix_full());
		F.reduce_parallel_rvu(params);

		VRUDecomp G(L.boundary_matrix_full());
		G.reduce_parallel_rvu(params);
	
		std::vector<int> new_order (number_cells_K);
		std::iota (new_order.begin(), new_order.end(), 0);


		std::sort(new_order.begin(), new_order.end(), [&](int i, int j) {
			//std::cerr << "comparing " << i << " and " << j;
			if (sorted_K_to_sorted_L[i] != -1 && sorted_K_to_sorted_L[j] != -1) {//FIXME: this needs to work with the sorted order. 
				//std::cerr << " which end up in 1" << std::endl;
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
			} else if (sorted_K_to_sorted_L[i] != -1 && sorted_K_to_sorted_L[j] == -1) {
				//std::cerr << " which end up in 2"<< std::endl;
				return true;
			} else if (sorted_K_to_sorted_L[i] == -1 && sorted_K_to_sorted_L[j] != -1) {
				//std::cerr << " which end up in 3"<< std::endl;
				return false;
			} else {
				//std::cerr << " which end up in 4"<< std::endl;
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

		std::cerr << "The new order is [";
		for (int i = 0; i < new_order.size(); i++) {
			std::cerr << " " << new_order[i];
		}
		std::cerr << "]" << std::endl;

		std::vector<int> old_to_new_order(number_cells_K);

		for (int i = 0; i < number_cells_K; i++) {
			old_to_new_order[new_order[i]] = i;
		}
		std::cerr << "old_to_new_order is [";
		for (int i = 0; i < number_cells_K; i++) {
			std::cerr << " " << old_to_new_order[i];
		}
		std::cerr << "]" << std::endl;
		MatrixData d_im;
		for (int i = 0; i < F.get_D().size(); i++) {
			std::vector<int> new_col_i;
			if (!F.get_D()[i].empty()) {
				for (int j = 0; j < F.get_D()[i].size(); j++) {
					new_col_i.push_back(old_to_new_order[F.get_D()[i][j]]);
				}
			}
			std::sort(new_col_i.begin(), new_col_i.end());
			std::cerr << "old column is [" ;
			for (int j = 0; j < F.get_D()[i].size(); j++) {
				std::cerr << " " << F.get_D()[i][j];
			}
			std::cerr << "] and new column is [";
			for (int j = 0; j < new_col_i.size(); j++) {
				std::cerr << " " << new_col_i[j];
			}
			std::cerr << "]" << std::endl;
			d_im.push_back(new_col_i);
		}
		MatrixData D_f = F.get_D();

		VRUDecomp Im(d_im);
		Im.reduce_parallel_rvu(params);

		MatrixData D_im = Im.get_D();
		MatrixData V_im = Im.get_V();
		Im.sanity_check();
		MatrixData V_f = F.get_V();
		std::cerr << "V_f is " << V_f << std::endl;

		std::cerr << "V_im is " << V_im << std::endl;

		std::vector<bool> to_keep(number_cells_K, false);
		for (int i = 0; i < V_f.size(); i++) {
			if (!V_f[i].empty()) {
				bool cycle = true;
				std::vector<int> quasi_sum (number_cells_K, 0);
				for (int j = 0; j < V_f[i].size(); j++) {
					for (int k = 0; k < D_f[V_f[i][j]].size(); k++) {
						quasi_sum[D_f[V_f[i][j]][k]]++;
					}
				}
				//std::cerr << "For " << i << " quasi_sum is [";
				//for (int j = 0; j < quasi_sum.size(); j++) {
				//	std::cerr << " " << quasi_sum[j];
				//}
				//std::cerr << "]" << std::endl;
				for (int j = 0; j < quasi_sum.size(); j++) {
					if (quasi_sum[j]%2 != 0) {
						cycle = false;
						break;
					}
				}
				to_keep[i] = cycle;
			}
		}

		MatrixData d_ker;

		std::vector<int> new_cols(number_cells_K, -1);
		int counter = 0;
		for (int i = 0; i < to_keep.size(); i++) {
			if (to_keep[i]) {
				d_ker.push_back(V_f[i]);
				new_cols[i] = counter;
				counter++;
			}
		}

		std::cerr << "new_cols is [";
		for (int i = 0; i < new_cols.size(); i++) {
			std::cerr << " " << new_cols[i];	
 		}
		std::cerr << "]" << std::endl;
		
		std::cerr << "D_ker is " << d_ker << std::endl;
		VRUDecomp Ker(d_ker, K.size());
		Ker.reduce_parallel_rvu(params);
		MatrixData R_ker = Ker.get_R();

		std::cerr << "R_ker is " << R_ker;

		MatrixData D_cok(F.get_D());
		MatrixData D_g = G.get_D();
		MatrixData R_g = G.get_R();
		MatrixData V_g = G.get_V();

		for (int i = 0; i < number_cells_L; i++) {
			bool replace = false;
			std::vector<int> quasi_sum (number_cells_L, 0);
			if (!V_g[i].empty()) {
				replace = true;
				for (int j = 0; j < V_g[i].size(); j++) {
					for (int k = 0; k < D_g[V_g[i][j]].size(); k++) {
						quasi_sum[D_g[V_g[i][j]][k]];//check if a column in V_g represents a cycle
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
				D_cok[sorted_L_to_sorted_K[i]] = V_g[i]; 
			} 
		}

		VRUDecomp Cok(D_cok);
		Cok.reduce_parallel_rvu(params);

		ImKerCokReduced<Int, Real> IKCR(K, L, F, G, Im, Ker, Cok, sorted_L_to_sorted_K, sorted_K_to_sorted_L, new_order);	
		
		IKCR.GenerateKerDiagrams(new_cols);
		IKCR.GenerateImDiagrams(new_cols);
		IKCR.GenerateCokDiagrams(new_cols);

		std::cerr << "F.get_R() is " << F.get_R() << std::endl;
		std::cerr << "F.get_D() is " << F.get_D() << std::endl;
		std::cerr << "F.get_V() is " << F.get_V() << std::endl;

		std::cerr << "G.get_R() is " << G.get_R() << std::endl;
		std::cerr << "G.get_D() is " << G.get_D() << std::endl;
		std::cerr << "G.get_V() is " << G.get_V() << std::endl;

		std::cerr << "Ker.get_R() is " << Ker.get_R() << std::endl;
		std::cerr << "Ker.get_D() is " << Ker.get_D() << std::endl;
		std::cerr << "Ker.get_V() is " << Ker.get_V() << std::endl;

		std::cerr << "Im.get_R() is " << Im.get_R() << std::endl;
		std::cerr << "Im.get_D() is " << Im.get_D() << std::endl;
		std::cerr << "Im.get_V() is " << Im.get_V() << std::endl;

		std::cerr << "Cok.get_R() is " << Cok.get_R() << std::endl;
		std::cerr << "Cok.get_D() is " << Cok.get_D() << std::endl;
		std::cerr << "Cok.get_V() is " << Cok.get_V() << std::endl;

		return  IKCR;
	}

}

