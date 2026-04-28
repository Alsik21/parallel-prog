


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <random>


#define TILE_SIZE 16
#define BLOCK_SIZE 16




__global__ void matmul_naive(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}


__global__ void matmul_tiled(const double* A, const double* B, double* C, int n) {
    __shared__ double A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ double B_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
       
        if (row < n && tile * TILE_SIZE + threadIdx.x < n) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_SIZE + threadIdx.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        if (col < n && tile * TILE_SIZE + threadIdx.y < n) {
            B_tile[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}


__global__ void matmul_coalesced(const double* A, const double* B, double* C, int n) {
    __shared__ double A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    double sum = 0.0;
    
    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        
        if (row < n && k * BLOCK_SIZE + tx < n) {
            A_shared[ty][tx] = A[row * n + k * BLOCK_SIZE + tx];
        } else {
            A_shared[ty][tx] = 0.0;
        }
        
        if (col < n && k * BLOCK_SIZE + ty < n) {
            B_shared[ty][tx] = B[(k * BLOCK_SIZE + ty) * n + col];
        } else {
            B_shared[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += A_shared[ty][i] * B_shared[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}



class Matrix {
private:
    std::vector<double> data;
    size_t n;

public:
    Matrix() : n(0) {}
    
    Matrix(size_t size) : n(size), data(size * size, 0.0) {}
    
    double& operator()(size_t i, size_t j) {
        return data[i * n + j];
    }
    
    const double& operator()(size_t i, size_t j) const {
        return data[i * n + j];
    }
    
    size_t size() const { return n; }
    size_t getSize() const { return n; }
    
    double* data_ptr() { return data.data(); }
    const double* data_ptr() const { return data.data(); }
    
    bool load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        
        std::vector<double> temp;
        double val;
        while (file >> val) temp.push_back(val);
        
        size_t loaded_n = static_cast<size_t>(std::sqrt(temp.size()));
        if (loaded_n * loaded_n != temp.size()) return false;
        
        n = loaded_n;
        data = temp;
        return true;
    }
    
    void save_to_file(const std::string& filename) const {
        std::ofstream file(filename);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                file << std::fixed << std::setprecision(6) << (*this)(i, j);
                if (j < n - 1) file << " ";
            }
            file << "\n";
        }
    }
    
    void randomize(double min_val = 0.0, double max_val = 100.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min_val, max_val);
        
        for (size_t i = 0; i < n * n; ++i) {
            data[i] = dist(gen);
        }
    }
    
    void fill_constant(double val) {
        std::fill(data.begin(), data.end(), val);
    }
};


Matrix sequential_multiply(const Matrix& A, const Matrix& B) {
    size_t n = A.size();
    Matrix C(n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = A(i, k);
            for (size_t j = 0; j < n; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    
    return C;
}



class CUDAMatrixMultiplier {
private:
    double *d_A, *d_B, *d_C;
    size_t n;
    bool initialized;
    
    void allocate_device_memory() {
        size_t size = n * n * sizeof(double);
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
    }
    
public:
    CUDAMatrixMultiplier() : d_A(nullptr), d_B(nullptr), d_C(nullptr), n(0), initialized(false) {}
    
    ~CUDAMatrixMultiplier() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    }
    
    void set_size(size_t size) {
        if (n != size) {
            if (initialized) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
            }
            n = size;
            allocate_device_memory();
            initialized = true;
        }
    }
    
    
    double multiply_naive(const Matrix& A, const Matrix& B, Matrix& C) {
        set_size(A.size());
        
        size_t size = n * n * sizeof(double);
        
        
        cudaMemcpy(d_A, A.data_ptr(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data_ptr(), size, cudaMemcpyHostToDevice);
        
        
        dim3 blockDim(16, 16);
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                     (n + blockDim.y - 1) / blockDim.y);
        
       
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
        
       
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        
        cudaMemcpy(C.data_ptr(), d_C, size, cudaMemcpyDeviceToHost);
        
        return elapsed_ms / 1000.0;
    }
    
   
    double multiply_tiled(const Matrix& A, const Matrix& B, Matrix& C) {
        set_size(A.size());
        
        size_t size = n * n * sizeof(double);
        
        cudaMemcpy(d_A, A.data_ptr(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data_ptr(), size, cudaMemcpyHostToDevice);
        
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, 
                     (n + TILE_SIZE - 1) / TILE_SIZE);
        
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cudaMemcpy(C.data_ptr(), d_C, size, cudaMemcpyDeviceToHost);
        
        return elapsed_ms / 1000.0;
    }
    

    double multiply_coalesced(const Matrix& A, const Matrix& B, Matrix& C) {
        set_size(A.size());
        
        size_t size = n * n * sizeof(double);
        
        cudaMemcpy(d_A, A.data_ptr(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data_ptr(), size, cudaMemcpyHostToDevice);
        
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                     (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matmul_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cudaMemcpy(C.data_ptr(), d_C, size, cudaMemcpyDeviceToHost);
        
        return elapsed_ms / 1000.0;
    }
    

    void benchmark_configurations(const Matrix& A, const Matrix& B, Matrix& C) {
        set_size(A.size());
        
        size_t size = n * n * sizeof(double);
        cudaMemcpy(d_A, A.data_ptr(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data_ptr(), size, cudaMemcpyHostToDevice);
        
        std::vector<std::pair<int, int>> block_sizes = {
            {8, 8}, {16, 16}, {32, 32}, {8, 16}, {16, 32}, {32, 16}
        };
        
        std::cout << "\n  Block Size Configuration Benchmark:" << std::endl;
        std::cout << "  -----------------------------------" << std::endl;
        std::cout << std::setw(12) << "Block (X,Y)" 
                  << std::setw(12) << "Grid (X,Y)" 
                  << std::setw(14) << "Time (s)" 
                  << std::setw(14) << "GFLOPS" << std::endl;
        
        for (auto& bs : block_sizes) {
            int bx = bs.first, by = bs.second;
            dim3 blockDim(bx, by);
            dim3 gridDim((n + bx - 1) / bx, (n + by - 1) / by);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
           
            matmul_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
            cudaDeviceSynchronize();
            
            cudaEventRecord(start);
            for (int iter = 0; iter < 10; ++iter) {
                matmul_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            double elapsed = elapsed_ms / 1000.0 / 10.0; // Average over 10 runs
            
            double gflops = (2.0 * n * n * n) / elapsed / 1e9;
            
            std::cout << std::setw(8) << bx << "x" << std::left << std::setw(3) << by 
                      << std::right << std::setw(10) << gridDim.x << "x" << gridDim.y
                      << std::setw(14) << std::fixed << std::setprecision(4) << elapsed
                      << std::setw(14) << std::setprecision(2) << gflops << std::endl;
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        
        cudaMemcpy(C.data_ptr(), d_C, size, cudaMemcpyDeviceToHost);
    }
};


void run_experiment(size_t n, int rank = 0) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "EXPERIMENT: n = " << n << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
   
    Matrix A(n), B(n), C_cpu(n), C_cuda_naive(n), C_cuda_tiled(n), C_cuda_coalesced(n);
    
   
    std::string a_file = "matrices/A_" + std::to_string(n) + ".txt";
    std::string b_file = "matrices/B_" + std::to_string(n) + ".txt";
    
    std::ifstream test_a(a_file);
    std::ifstream test_b(b_file);
    
    if (test_a.good() && test_b.good()) {
        A.load_from_file(a_file);
        B.load_from_file(b_file);
        std::cout << "Loaded matrices from files" << std::endl;
    } else {
        std::cout << "Generating random matrices..." << std::endl;
        A.randomize();
        B.randomize();
        A.save_to_file(a_file);
        B.save_to_file(b_file);
    }
    
   
    std::cout << "\n--- CPU Sequential Multiplication ---" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    C_cpu = sequential_multiply(A, B);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();
    
    double cpu_gflops = (2.0 * n * n * n) / cpu_time / 1e9;
    std::cout << "Time: " << cpu_time << " s" << std::endl;
    std::cout << "Performance: " << cpu_gflops << " GFLOPS" << std::endl;
    
    
    CUDAMatrixMultiplier cuda_mm;
    
    std::cout << "\n--- CUDA Naive Multiplication ---" << std::endl;
    double cuda_naive_time = cuda_mm.multiply_naive(A, B, C_cuda_naive);
    double cuda_naive_gflops = (2.0 * n * n * n) / cuda_naive_time / 1e9;
    std::cout << "Time: " << cuda_naive_time << " s" << std::endl;
    std::cout << "Performance: " << cuda_naive_gflops << " GFLOPS" << std::endl;
    
    std::cout << "\n--- CUDA Tiled Multiplication (Tile Size = " << TILE_SIZE << ") ---" << std::endl;
    double cuda_tiled_time = cuda_mm.multiply_tiled(A, B, C_cuda_tiled);
    double cuda_tiled_gflops = (2.0 * n * n * n) / cuda_tiled_time / 1e9;
    std::cout << "Time: " << cuda_tiled_time << " s" << std::endl;
    std::cout << "Performance: " << cuda_tiled_gflops << " GFLOPS" << std::endl;
    
    std::cout << "\n--- CUDA Coalesced Multiplication ---" << std::endl;
    double cuda_coalesced_time = cuda_mm.multiply_coalesced(A, B, C_cuda_coalesced);
    double cuda_coalesced_gflops = (2.0 * n * n * n) / cuda_coalesced_time / 1e9;
    std::cout << "Time: " << cuda_coalesced_time << " s" << std::endl;
    std::cout << "Performance: " << cuda_coalesced_gflops << " GFLOPS" << std::endl;
    
  
    std::cout << "\n--- CUDA Configuration Benchmark ---" << std::endl;
    Matrix C_bench(n);
    cuda_mm.benchmark_configurations(A, B, C_bench);
    
   
    std::cout << "\n--- Verification ---" << std::endl;
    double max_diff_naive = 0.0, max_diff_tiled = 0.0, max_diff_coalesced = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            max_diff_naive = std::max(max_diff_naive, std::abs(C_cpu(i, j) - C_cuda_naive(i, j)));
            max_diff_tiled = std::max(max_diff_tiled, std::abs(C_cpu(i, j) - C_cuda_tiled(i, j)));
            max_diff_coalesced = std::max(max_diff_coalesced, std::abs(C_cpu(i, j) - C_cuda_coalesced(i, j)));
        }
    }
    
    std::cout << "Naive max difference: " << std::scientific << max_diff_naive;
    std::cout << (max_diff_naive < 1e-9 ? " ✓" : " ✗") << std::endl;
    std::cout << "Tiled max difference: " << max_diff_tiled;
    std::cout << (max_diff_tiled < 1e-9 ? " ✓" : " ✗") << std::endl;
    std::cout << "Coalesced max difference: " << max_diff_coalesced;
    std::cout << (max_diff_coalesced < 1e-9 ? " ✓" : " ✗") << std::endl;
    
   
    C_cuda_coalesced.save_to_file("results/C_" + std::to_string(n) + ".txt");
    
  
    std::cout << "\n--- Speedup Summary ---" << std::endl;
    std::cout << "Naive vs CPU: " << cpu_time / cuda_naive_time << "x" << std::endl;
    std::cout << "Tiled vs CPU: " << cpu_time / cuda_tiled_time << "x" << std::endl;
    std::cout << "Coalesced vs CPU: " << cpu_time / cuda_coalesced_time << "x" << std::endl;
}



int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "MATRIX MULTIPLICATION WITH CUDA" << std::endl;
    std::cout << "============================================================" << std::endl;
    
   
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "\nGPU Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  CUDA Cores: " << prop.multiProcessorCount * prop.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
    
    
    system("mkdir -p matrices results");
    
    
    std::vector<size_t> matrix_sizes = {500, 1000, 1600, 2000, 2500};
    
    
  
    
  
    for (size_t n : matrix_sizes) {
      
        size_t memory_needed = 3 * n * n * sizeof(double);
        if (memory_needed > prop.totalGlobalMem * 0.8) {
            std::cout << "\nSkipping n=" << n << " (insufficient GPU memory)" << std::endl;
            continue;
        }
        
        run_experiment(n);
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ALL EXPERIMENTS COMPLETED" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}