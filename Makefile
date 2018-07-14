CXX=g++

OPT = -I lib -O2 -mcmodel=medium  -fopenmp -w 

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: GTA

GTA: GTA.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $<

demo: GTA.cpp
	g++ -I lib -o GTA GTA.cpp -O2 -fopenmp -w -mcmodel=medium
	./GTA sample/input.txt sample/result 3 10 20


.PHONY: clean

clean:
	rm -f GTA

