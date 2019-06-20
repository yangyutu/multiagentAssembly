CC = gcc
CXX = g++
VPATH=testParticleSimulator


HOME=/home/yangyutu/
BOOST_INCLUDE=-I/opt/boost/boost_1_57_0


DEBUGFLAG=-DDEBUG -g3 -O0 -fPIC
RELEASEFLAG= -O3 -march=native -DARMA_NO_DEBUG
CXXFLAGS=  -std=c++11 $(BOOST_INCLUDE) -D__LINUX -fopenmp  `python-config --cflags` `/home/yangyutu/anaconda3/bin/python -m pybind11 --includes` 
LDFLAG= -L/opt/OpenBLAS/lib  -llapack -lblas -fopenmp -lpthread -pthread -no-pie `python-config --ldflags`

OBJ=testSimulator.o ParticleSimulator.o GridWorld.o ParallelNormalReservoir.o omprng.o rngstream.o

test.exe: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAG) 
	
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $(RELEASEFLAG) $^
	




clean:
	rm *.o *.exe
	
