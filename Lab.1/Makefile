CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall
TARGET = matrix_mult.exe
SOURCES = matrix_mult.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

generate:
	python generate_matrices.py 5

run: $(TARGET) generate
	./$(TARGET)
	python verify.py

clean:
	rm -f $(TARGET) *.txt

list:
	ls -la *.txt

show-time:
	cat execution_time.txt

test-small: clean
	python generate_matrices.py 10
	./$(TARGET)
	python verify.py
	cat execution_time.txt

test-medium: clean
	python generate_matrices.py 100
	./$(TARGET)
	python verify.py
	cat execution_time.txt

test-large: clean
	python generate_matrices.py 500
	./$(TARGET)
	python verify.py
	cat execution_time.txt

.PHONY: all generate run clean list show-time test-small test-medium test-large