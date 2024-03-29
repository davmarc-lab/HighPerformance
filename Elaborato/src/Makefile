# Default makefile for the circles intersections program.
#
# Available targets:
#
# - make
#   builds the standard executable
#
# - make clean
#   remove all output files and executables
#
# - make movie
#   compile the executable circles.movie that writes a gnuplot
#   file at each iteration; the executable is run and the output
#   is processed to produce an animation circles.avi

# Last modified 2023-12-06 by Moreno Marzolla

EXE:=circles
CFLAGS=-std=c99 -Wall -Wpedantic
LDLIBS+=-lm

ALL: $(EXE)

$(EXE).movie: CFLAGS+=-DMOVIE
$(EXE).movie: $(EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

movie: $(EXE).movie
	./$(EXE).movie 200 500
	for f in *.gp; do echo "Processing $$f"; gnuplot "$$f"; done
	ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi

# - make MP
#   to compile the openmp version.

MP: omp-$(EXE)

omp-$(EXE): CFLAGS+= -fopenmp
omp-$(EXE): omp-$(EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# - make CUDA
#   to compile the cuda version.

CUDA: cuda-$(EXE)

cuda-$(EXE): CFLAGS=
cuda-$(EXE): cuda-$(EXE).cu
	nvcc $(CFLAGS) $< -o $@ $(LDLIBS)


clean:
	rm -f $(EXE) omp-$(EXE) cuda-$(EXE) $(EXE).movie *.o *~ *.gp *.png *.avi

