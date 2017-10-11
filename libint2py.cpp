#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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
        libint2::finalize();
        PyList_Append(PyBundle, S);
        PyList_Append(PyBundle, T);
        PyList_Append(PyBundle, V);
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
      
      
      
      
      
      
      
      
