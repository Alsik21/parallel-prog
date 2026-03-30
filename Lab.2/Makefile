CC = g++
CFLAGS = -fopenmp -O2 -Wall
LDFLAGS = -lgomp
TARGET = matrix_mult_omp.exe
SOURCES = matrix_mult_omp.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

test: $(TARGET)
	python generate_matrices.py 200
	./$(TARGET) 1
	./$(TARGET) 2
	./$(TARGET) 4
	./$(TARGET) 8

clean:
	del -f $(TARGET) *.txt output_*.txt
	rmdir /s /q results 2>nul

.PHONY: all test clean