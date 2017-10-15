#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Python.h>
#include <libint2.hpp>
extern "C"{
    using namespace std;
    using namespace libint2;

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                     // this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
                     // to meet the layout of the integrals returned by the Libint integral library

    PyObject* compute_1body_ints(const std::vector<libint2::Shell>& shells,
                              libint2::Operator t,
                              const std::vector<Atom>& atoms = std::vector<Atom>());
    PyObject* compute_2body_ints(const std::vector<libint2::Shell>& shells);

    size_t nbasis(const std::vector<libint2::Shell>& shells);

    vector<Atom> read_py_atoms(PyObject* py_atoms);

	PyObject* integrals(PyObject* py_atoms, const char* py_basis){

		//string filename="h2o.xyz";
		//cout << filename << endl;
		//ifstream input_file(filename);
		//vector<Atom> atoms = read_dotxyz(input_file);
        vector<Atom> atoms = read_py_atoms(py_atoms);
		BasisSet obs(py_basis, atoms);
        
        PyObject* PyBundle = PyList_New(0);
        libint2::initialize();
        // compute overlap integrals
        auto S = compute_1body_ints(obs, Operator::overlap);
        // compute kinetic-energy integrals
        auto T = compute_1body_ints(obs, Operator::kinetic);
        // compute nuclear-attraction integrals
        auto V = compute_1body_ints(obs, Operator::nuclear, atoms);
        // compute eri
        auto G = compute_2body_ints(obs);
        libint2::finalize();
        PyList_Append(PyBundle, S);
        PyList_Append(PyBundle, T);
        PyList_Append(PyBundle, V);
        PyList_Append(PyBundle, G);
        return PyBundle;
	}

    size_t nbasis(const std::vector<libint2::Shell>& shells) {
        size_t n = 0;
        for (const auto& shell: shells)
        n += shell.size();
        return n;
    }

    size_t max_nprim(const std::vector<libint2::Shell>& shells) {
        size_t n = 0;
        for (auto shell: shells)
        n = std::max(shell.nprim(), n);
        return n;
    }

    int max_l(const std::vector<libint2::Shell>& shells) {
        int l = 0;
        for (auto shell: shells)
            for (auto c: shell.contr)
                l = std::max(c.l, l);
        return l;
    }

    std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
        std::vector<size_t> result;
        result.reserve(shells.size());

        size_t n = 0;
        for (auto shell: shells) {
            result.push_back(n);
            n += shell.size();
        }

        return result;
    }


    PyObject* compute_1body_ints(const std::vector<libint2::Shell>& shells,
                              libint2::Operator obtype,
                              const std::vector<Atom>& atoms)
    {
        using libint2::Shell;
        using libint2::Engine;
        using libint2::Operator;

        const auto n = nbasis(shells);
        Matrix result(n,n);

        // construct the overlap integrals engine
        Engine engine(obtype, max_nprim(shells), max_l(shells), 0);
        // nuclear attraction ints engine needs to know where the charges sit ...
        // the nuclei are charges in this case; in QM/MM there will also be classical charges
        if (obtype == Operator::nuclear) {
            std::vector<std::pair<double,std::array<double,3>>> q;
            for(const auto& atom : atoms) {
                q.push_back( {static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
            }
            engine.set_params(q);
        }

        auto shell2bf = map_shell_to_basis_function(shells);

        // buf[0] points to the target shell set after every call  to engine.compute()
        const auto& buf = engine.results();

        // loop over unique shell pairs, {s1,s2} such that s1 >= s2
        // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
        for(auto s1=0; s1!=shells.size(); ++s1) {

            auto bf1 = shell2bf[s1]; // first basis function in this shell
            auto n1 = shells[s1].size();

            for(auto s2=0; s2<=s1; ++s2) {

                auto bf2 = shell2bf[s2];
                auto n2 = shells[s2].size();

                // compute shell pair; return is the pointer to the buffer
                engine.compute(shells[s1], shells[s2]);

                // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
                Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
                result.block(bf1, bf2, n1, n2) = buf_mat;
                if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
                result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

              }
        }

        PyObject* PyResult = PyList_New(0);
        for(auto i=0;i<n;i++)
            for(auto j=0;j<n;j++)
                PyList_Append(PyResult,PyFloat_FromDouble(result(i,j)));
        return PyResult;
    }
    
      
    PyObject* compute_2body_ints(const std::vector<libint2::Shell>& shells){
    
      using libint2::Shell;
      using libint2::Engine;
      using libint2::Operator;
    
      const auto n = nbasis(shells);
      //Matrix G = Matrix::Zero(n,n);
    
      // construct the 2-electron repulsion integrals engine
      Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
    
      auto shell2bf = map_shell_to_basis_function(shells);
    
      const auto& buf = engine.results();
      PyObject* PyResult = PyList_New(0);
    
      // The problem with the simple Fock builder is that permutational symmetries of the Fock,
      // density, and two-electron integrals are not taken into account to reduce the cost.
      // To make the simple Fock builder efficient we must rearrange our computation.
      // The most expensive step in Fock matrix construction is the evaluation of 2-e integrals;
      // hence we must minimize the number of computed integrals by taking advantage of their permutational
      // symmetry. Due to the multiplicative and Hermitian nature of the Coulomb kernel (and realness
      // of the Gaussians) the permutational symmetry of the 2-e ints is given by the following relations:
      //
      // (12|34) = (21|34) = (12|43) = (21|43) = (34|12) = (43|12) = (34|21) = (43|21)
      //
      // (here we use chemists' notation for the integrals, i.e in (ab|cd) a and b correspond to
      // electron 1, and c and d -- to electron 2).
      //
      // It is easy to verify that the following set of nested loops produces a permutationally-unique
      // set of integrals:
      // foreach a = 0 .. n-1
      //   foreach b = 0 .. a
      //     foreach c = 0 .. a
      //       foreach d = 0 .. (a == c ? b : c)
      //         compute (ab|cd)
      //
      // The only complication is that we must compute integrals over shells. But it's not that complicated ...
      //
      // The real trick is figuring out to which matrix elements of the Fock matrix each permutationally-unique
      // (ab|cd) contributes. STOP READING and try to figure it out yourself. (to check your answer see below)
    
      // loop over permutationally-unique set of shells
      const auto n_unique_eri = n + n*(n-1)*2 + n*(n-1)*(n-2) + n*(n-1)*(n-2)*(n-3)/8;
      //cout<<n<<"++++++++++++++++++++++++++++++++"<<endl;
      //cout << n_unique_eri <<"++++++++++++++++++++++++++++++++"<<endl;
      struct eri_pair{
	double value;
  	unsigned long long weigh;
      };	
      eri_pair eri_store[n_unique_eri];
      int ctr = 0;
      for(auto s1=0; s1!=shells.size(); ++s1) {
    
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = shells[s1].size();   // number of basis functions in this shell
    
        for(auto s2=0; s2<=s1; ++s2) {
    
          auto bf2_first = shell2bf[s2];
          auto n2 = shells[s2].size();
    
          for(auto s3=0; s3<=s1; ++s3) {
    
            auto bf3_first = shell2bf[s3];
            auto n3 = shells[s3].size();
    
            for(auto s4=0; s4<=s3; ++s4) {
    
              auto bf4_first = shell2bf[s4];
              auto n4 = shells[s4].size();
    
              // compute the permutational degeneracy (i.e. # of equivalents) of the given shell set
              //auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
              //auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
              //auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
              //auto s1234_deg = s12_deg * s34_deg * s12_34_deg;
    
              //const auto tstart = std::chrono::high_resolution_clock::now();
    
              engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
              const auto* buf_1234 = buf[0];
              if (buf_1234 == nullptr)
                continue; // if all integrals screened out, skip to next quartet
    
              //const auto tstop = std::chrono::high_resolution_clock::now();
              //time_elapsed += tstop - tstart;
    
              // ANSWER
              // 1) each shell set of integrals contributes up to 6 shell sets of the Fock matrix:
              //    F(a,b) += (ab|cd) * D(c,d)
              //    F(c,d) += (ab|cd) * D(a,b)
              //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
              //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
              //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
              //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
              // 2) each permutationally-unique integral (shell set) must be scaled by its degeneracy,
              //    i.e. the number of the integrals/sets equivalent to it
              // 3) the end result must be symmetrized
              for(auto f1=0, f1234=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2) {
                  for(auto f3=0; f3!=n3; ++f3) {
                    for(auto f4=0; f4!=n4; ++f4, ++f1234) {
			    if(bf1_first+f1 < bf2_first+f2)
				    continue;
			    if(bf3_first+f3 < bf4_first+f4)
				    continue;
			    if(bf1_first+f1 < bf3_first+f3)
				    continue;
			    if(bf1_first+f1 == bf3_first+f3 and bf2_first+f2 < bf4_first+f4)
				    continue;
   		      //cout << " ( " << bf1_first+f1 << " " << bf2_first+f2 << " | " << bf3_first+f3 << " " << bf4_first+f4 << " ) = " << buf_1234[((f1*n2+f2)*n3+f3)*n4+f4] << endl;
		      eri_store[ctr].value = buf_1234[f1234];
		      eri_store[ctr].weigh = (((bf1_first+f1)*n + bf2_first+f2)*n+ bf3_first+f3)*n + bf4_first+f4;
		      ctr ++;
                    }
                  }
                }
              }
	      //cout<< "ctr = " << ctr << endl;
            }
          }
        }
      }
      sort(eri_store,eri_store+n_unique_eri,[](eri_pair a, eri_pair b)->bool{return a.weigh < b.weigh;});

      for(int i=0;i<n_unique_eri;i++)
	      PyList_Append(PyResult,PyFloat_FromDouble(eri_store[i].value));
    
      return PyResult;
    }
    
    vector<Atom> read_py_atoms(PyObject* py_molecule){
        // In libint the default unit for length is Bohr, e.g. a.u.
        // but the commonly used unit for length is angstrom.
        // Therefore we have to pay attention to converting the unit when reading the coordinates.
        // Here we assume that the python program has converted all length to Bohr.
        // the 2010 CODATA reference set, available at DOI 10.1103/RevModPhys.84.1527
        //const double bohr_to_angstrom = 0.52917721092;

        vector<Atom> atoms;
        int n = PyList_Size(py_molecule);

        //cout << "# Atoms = " << n << endl;
        for (int i = 0; i < n; i++){
            PyObject* py_atom = PyList_GetItem(py_molecule,i);
            Atom atom;
            atom.atomic_number = (int)PyLong_AsLong(PyList_GetItem(py_atom,0));
            atom.x = PyFloat_AsDouble(PyList_GetItem(py_atom,1));
            atom.y = PyFloat_AsDouble(PyList_GetItem(py_atom,2));
            atom.z = PyFloat_AsDouble(PyList_GetItem(py_atom,3));
            //cout << atom.atomic_number << "\t" << atom.x << "\t\t" << atom.y << "\t\t"  << atom.z << endl;
            atoms.push_back(atom);
        }
        return atoms;
    }
}     
      
      
      
      
      
      
      
      
