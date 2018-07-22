CXX=g++

OPT = -I lib -O2 -lOpenCL  -mcmodel=medium  -fopenmp -w 

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: demo

GTA: src/GTA.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $<

demo: src/GTA.cpp
	$(CXX) $(CXXFLAGS)  -o $@  $<
	./demo.sh


.PHONY: clean

clean:
	rm -f GTA demo

