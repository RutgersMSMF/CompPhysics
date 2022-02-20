CC = g++-10
PYTHON = -IC:\Users\steve\AppData\Local\Programs\Python\Python39\Include
PYBIND = -IC:\Users\Steve\AppData\Local\Programs\Python\Python39\lib\site-packages\pybind11\include
LDFLAGS = -LC:/msys64/mingw64/lib -lpython3.9  -lm -lversion -lshlwapi -lm

tricorn.pyd : tricorn.cc
	g++ $(PYTHON) $(PYBIND) -O3 -fopenmp -shared -std=c++11 -fPIC tricorn.cc -o tricorn.pyd $(LDFLAGS)

clean :
	del tricorn.pyd
