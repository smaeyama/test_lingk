##
# Makefile
#                            by S. Maeyama (Apr. 2011)
#

### Intel fortran ###
FC = ifx
FFLAGS = -warn all -fpp -traceback \
         -g -check bound -fpe0 #-check uninit
FFLAGS += -ipo -O3
INC =
LIB = 
OPENMP = #-qopenmp

#### GNU fortran ###
#FC = gfortran
#FFLAGS = #-Wall -Wextra -pedantic -fbacktrace \
#         -fbounds-check -Wuninitialized -ffpe-trap=invalid,zero,overflow 
#FFLAGS += -mcmodel=medium -m64 -march=native -mtune=native -O3 #-ffast-math
#INC = 
#LIB = 
#OPENMP = #-fopenmp #-Wl,--stack,2048000000


### create lingk.exe ###
lingk:	./src/parameters.f90\
	./src/functions.f90\
	./src/geometry.f90\
	./src/fileio.f90\
	./src/clock.f90\
	./src/lingk.f90

	$(FC) $(FFLAGS) $(OPENMP) -c ./src/parameters.f90
	$(FC) $(FFLAGS) $(OPENMP) -c ./src/functions.f90
	$(FC) $(FFLAGS) $(OPENMP) -c ./src/geometry.f90
	$(FC) $(FFLAGS) $(OPENMP) -c ./src/fileio.f90
	$(FC) $(FFLAGS) $(OPENMP) -c ./src/clock.f90
	$(FC) $(FFLAGS) $(OPENMP) -c ./src/lingk.f90

	$(FC) $(FFLAGS) $(OPENMP) *.o -o lingk.exe $(INC) $(LIB)

	rm -f *.o *.mod *__genmod*


### clean up files ###
clean:
	rm -f *.o *.mod *.exe

clear:
	rm -f ./data/* ./*.err ./*.out *.q.o* *.q.e*

