_DEPS = traj.h traj_physics.h
_OBJ = utils.o TS3.o tnp.o generate.o generaterandp.o  save.o cl_code.o changedt.o calcEBV_FFT.o calcU.o  get_densityfields.o
#sel_part_print.o smoothfield.o calc_trilin_constants.o
IDIR = include
dir_guard=@mkdir -p $(@D)
#https://stackoverflow.com/questions/14492436/g-optimization-beyond-o3-ofast
CC=g++
#ucrt64
CFLAGS= -g -I$(IDIR) -I /ucrt64/include/vtk -L /ucrt64/lib/vtk -fopenmp -fopenmp-simd -march=native -malign-double -std=c++2b 
#CFLAGS= -g -I$(IDIR) -I /ucrt64/include/vtk -L /ucrt64/lib/vtk -march=native -malign-double -std=c++2b 
#CFLAGS= -g -I$(IDIR) -I /usr/include/vtk -L /usr/lib/x86_64-linux-gnu -fopenmp -fopenmp-simd -march=native -malign-double -std=c++2b 

CFLAGS+= -Ofast -ftree-parallelize-loops=8 
CFLAGS+= -mavx -mavx2 -mfma -ffast-math -ftree-vectorize -fomit-frame-pointer
#mingw64
#CFLAGS= -I$(IDIR) -I /mingw64/include/vtk -L /mingw64/lib/vtk -fopenmp -fopenmp-simd -Ofast -march=native -malign-double -ftree-parallelize-loops=8 -std=c++2b
#CFLAGS= -I$(IDIR) -fopenmp -fopenmp-simd -Ofast -march=native -malign-double -ftree-parallelize-loops=8 -std=c++2b
#CFLAGSd= -g -I$(IDIR) -I /ucrt64/include/vtk-9.1 -L /ucrt64/lib/vtk -fopenmp -fopenmp-simd -march=native -malign-double -std=c++2b 
#LIBS= -lm -lgsl -lOpenCL.dll -lfftw3f -lomp.dll -lfftw3f_omp
#LIBS= -lm -lgsl -lOpenCL.dll  -lgomp.dll  -lfftw3f_omp -lfftw3f
LIBS= -lm -lgsl -lOpenCL  -lgomp  -lfftw3f_omp -lfftw3f
#-lnfft3f.dll -lnfft3f_threads
#-lnfft3-4.dll 
LIBS+= -lvtkCommonCore.dll  -lvtksys.dll -lvtkIOXML.dll -lvtkCommonDataModel.dll -lvtkIOCore.dll
#LIBS+= -lvtkCommonCore-9.1 -lvtksys-9.1 -lvtkIOXML-9.1 -lvtkCommonDataModel-9.1 -lvtkIOCore-9.1

#-lvtkIOLegacy.dll -lvtkCommonComputationalGeometry.dll -lvtkCommonSystem.dll
#-lvtkGraphics.dll -lvtkFiltersGeneral.dll -lvtkImagingCore.dll -lvtkFiltersGeneric.dll -lvtkIOCore.dll -lvtkIOImage.dll 
AFLAGS= -flto -funroll-loops -fno-signed-zeros -fno-trapping-math -D_GLIBCXX_PARALLEL -fgcse-sm -fgcse-las 
#-Wl,--stack,4294967296

#CC=clang++
#CFLAGS=-I$(IDIR) -fopenmp -fopenmp-simd -O3 -Ofast -mavx -mfma -ffast-math -ftree-vectorize -march=native -fomit-frame-pointer -malign-double -std=c++2b
#AFLAGS=
#LIBS=-lm -lgsl -lOpenCL -lfftw3f -lomp.dll

CFLAGS += $(AFLAGS)
CPUS ?= $(shell (nproc --all || sysctl -n hw.ncpu) 2>/dev/null || echo 1)
MAKEFLAGS += --jobs=$(CPUS)

ODIR=obj
DODIR=obj_debug
LDIR=lib

DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
DOBJ = $(patsubst %,$(DODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(dir_guard)
	$(CC) -c -o $@ $< $(CFLAGS)

$(DODIR)/%.o: %.cpp $(DEPS)
	$(dir_guard)
	$(CC) -g -c -o $@ $< $(CFLAGSd)

TS3: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

debug: $(DOBJ)
	$(CC) -v -o TS3$@ $^ $(CFLAGSd) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(DODIR)/*.o *~ core $(INCDIR)/*~ TS3.exe TS3debug.exe *.vti *.vtp TS3