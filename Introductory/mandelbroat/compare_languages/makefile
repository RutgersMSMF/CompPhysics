CC = g++-10
F90 = gfortran

all : mandc mandf

mandc : mandc.cc
	g++ -O3 -fopenmp -o mandc mandc.cc

mandf : mandf.f90
	gfortran -O3 -fopenmp -o mandf mandf.f90

clean :
	del -f mandc mandf

