LIBINT_INSTALL_PATH :=/usr/local/libint/2.2.0
LIBINT_BASIS_PATH :=\"/usr/local/libint/2.2.0/share/libint/2.2.0/basis\"
EIGEN_INSTALL_PATH :=/root/Software/libint/eigen
PYTHON_INCLUDE_PATH :=/usr/include/python3.5

CXX := g++ -std=c++11

all: libint2py.cpp
	$(CXX) -fPIC -shared -o libint2py.so libint2py.cpp -lpython3.5m -lint2 -DSRCDATADIR=$(LIBINT_BASIS_PATH) -I$(LIBINT_INSTALL_PATH)/include -I$(LIBINT_INSTALL_PATH)/include/libint2 -I$(EIGEN_INSTALL_PATH) -I$(PYTHON_INCLUDE_PATH)

	
