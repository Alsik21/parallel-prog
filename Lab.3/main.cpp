// matrix_multiply.cpp
// Лабораторная работа №1: Перемножение квадратных матриц с MPI

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <mpi.h>

class Matrix {
private:
    std::vector<double> data;
    size_t n;

public:
    Matrix() : n(0) {}
    
    Matrix(size_t size) : n(size), data(size * size, 0.0) {}
    
    Matrix(size_t rows, size_t cols) : n(rows), data(rows * cols, 0.0) {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square for multiplication");
        }
    }
    
    double& operator()(size_t i, size_t j) {
        return data[i * n + j];
    }
    
    const double& operator()(size_t i, size_t j) const {
        return data[i * n + j];
    }
    
    size_t size() const { return n; }
    
    double* data_ptr() { return data.data(); }
    const double* data_ptr() const { return data.data(); }
    
    // Загрузка матрицы из файла
    bool load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::vector<double> temp;
        double val;
        while (file >> val) {
            temp.push_back(val);
        }
        
        size_t loaded_n = static_cast<size_t>(std::sqrt(temp.size()));
        if (loaded_n * loaded_n != temp.size()) {
            std::cerr << "Error: File data does not form a square matrix\n";
            return false;
        }
        
        n = loaded_n;
        data = temp;
        return true;
    }
    
    // Сохранение матрицы в файл
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
    
    // Генерация случайной матрицы
    void randomize(double min_val = 0.0, double max_val = 100.0) {
        for (size_t i = 0; i < n * n; ++i) {
            data[i] = min_val + static_cast<double>(rand()) / RAND_MAX * (max_val - min_val);
        }
    }
};

// Последовательное умножение матриц (для верификации)
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

// Параллельное умножение матриц с использованием MPI
// Используется распределение строк матрицы A между процессами
class ParallelMatrixMultiplier {
private:
    MPI_Comm comm;
    int rank, size;
    
    // Распределение строк между процессами
    void get_row_distribution(size_t n, int& local_rows, int& start_row) const {
        int rows_per_proc = n / size;
        int remainder = n % size;
        
        if (rank < remainder) {
            local_rows = rows_per_proc + 1;
            start_row = rank * (rows_per_proc + 1);
        } else {
            local_rows = rows_per_proc;
            start_row = rank * rows_per_proc + remainder;
        }
    }
    
public:
    ParallelMatrixMultiplier(MPI_Comm comm_ = MPI_COMM_WORLD) : comm(comm_) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }
    
    // Параллельное умножение
    Matrix multiply(const Matrix& A, const Matrix& B) {
        size_t n = A.size();
        Matrix C(n);
        
        // Тайминг только на процессе 0
        double local_start = 0.0, local_end = 0.0;
        if (rank == 0) {
            local_start = MPI_Wtime();
        }
        
        // 1. Рассылаем матрицу B всем процессам
        MPI_Bcast(const_cast<double*>(B.data_ptr()), n * n, MPI_DOUBLE, 0, comm);
        
        // 2. Определяем, какие строки A будет обрабатывать каждый процесс
        int local_rows, start_row;
        get_row_distribution(n, local_rows, start_row);
        
        // 3. Распределяем строки A
        std::vector<double> A_local(local_rows * n);
        
        if (rank == 0) {
            // Отправляем каждому процессу его строки
            for (int dest = 1; dest < size; ++dest) {
                int dest_rows, dest_start;
                // Временный объект для вычисления распределения
                ParallelMatrixMultiplier temp(comm);
                // Нужно переключить rank
                int old_rank = rank;
                // Эмуляция: вычисляем распределение для dest
                int rows_per_proc = n / size;
                int rem = n % size;
                if (dest < rem) {
                    dest_rows = rows_per_proc + 1;
                    dest_start = dest * (rows_per_proc + 1);
                } else {
                    dest_rows = rows_per_proc;
                    dest_start = dest * rows_per_proc + rem;
                }
                
                std::vector<double> dest_data(dest_rows * n);
                for (int i = 0; i < dest_rows; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        dest_data[i * n + j] = A(dest_start + i, j);
                    }
                }
                MPI_Send(dest_data.data(), dest_rows * n, MPI_DOUBLE, dest, 0, comm);
            }
            
            // Копируем строки для процесса 0
            for (int i = 0; i < local_rows; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    A_local[i * n + j] = A(start_row + i, j);
                }
            }
        } else {
            // Получаем свои строки
            MPI_Recv(A_local.data(), local_rows * n, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
        }
        
        // 4. Локальное умножение
        std::vector<double> C_local(local_rows * n, 0.0);
        
        for (int i = 0; i < local_rows; ++i) {
            for (size_t k = 0; k < n; ++k) {
                double aik = A_local[i * n + k];
                for (size_t j = 0; j < n; ++j) {
                    C_local[i * n + j] += aik * B(k, j);
                }
            }
        }
        
        // 5. Сбор результатов
        // Вычисляем смещения для MPI_Gatherv
        std::vector<int> recv_counts(size);
        std::vector<int> displs(size);
        
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            int p_rows;
            int p_start;
            // Вычисляем для процесса p
            int rows_per_proc = n / size;
            int rem = n % size;
            if (p < rem) {
                p_rows = rows_per_proc + 1;
            } else {
                p_rows = rows_per_proc;
            }
            recv_counts[p] = p_rows * n;
            displs[p] = offset;
            offset += recv_counts[p];
        }
        
        std::vector<double> C_full(n * n);
        
        MPI_Gatherv(C_local.data(), local_rows * n, MPI_DOUBLE,
                    C_full.data(), recv_counts.data(), displs.data(),
                    MPI_DOUBLE, 0, comm);
        
        // 6. Формируем результирующую матрицу на процессе 0
        if (rank == 0) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    C(i, j) = C_full[i * n + j];
                }
            }
            
            local_end = MPI_Wtime();
            std::cout << "Parallel multiplication time: " << (local_end - local_start) << " seconds\n";
        }
        
        return C;
    }
    
    // Альтернативная реализация с использованием MPI_Scatterv/MPI_Gatherv
    Matrix multiply_scatter(const Matrix& A, const Matrix& B) {
        size_t n = A.size();
        Matrix C(n);
        
        double total_start = MPI_Wtime();
        
        // Рассылаем B
        MPI_Bcast(const_cast<double*>(B.data_ptr()), n * n, MPI_DOUBLE, 0, comm);
        
        // Подготавливаем распределение строк
        int rows_per_proc = n / size;
        int remainder = n % size;
        
        std::vector<int> send_counts(size);
        std::vector<int> displs(size);
        
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            if (p < remainder) {
                send_counts[p] = (rows_per_proc + 1) * n;
            } else {
                send_counts[p] = rows_per_proc * n;
            }
            displs[p] = offset;
            offset += send_counts[p];
        }
        
        int local_rows = send_counts[rank] / n;
        std::vector<double> A_local(local_rows * n);
        
        // Распределяем строки A
        if (rank == 0) {
            std::vector<double> A_full(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    A_full[i * n + j] = A(i, j);
                }
            }
            MPI_Scatterv(A_full.data(), send_counts.data(), displs.data(),
                        MPI_DOUBLE, A_local.data(), local_rows * n, MPI_DOUBLE, 0, comm);
        } else {
            MPI_Scatterv(nullptr, send_counts.data(), displs.data(),
                        MPI_DOUBLE, A_local.data(), local_rows * n, MPI_DOUBLE, 0, comm);
        }
        
        // Локальное умножение
        std::vector<double> C_local(local_rows * n, 0.0);
        
        for (int i = 0; i < local_rows; ++i) {
            for (size_t k = 0; k < n; ++k) {
                double aik = A_local[i * n + k];
                for (size_t j = 0; j < n; ++j) {
                    C_local[i * n + j] += aik * B(k, j);
                }
            }
        }
        
        // Сбор результатов
        if (rank == 0) {
            std::vector<double> C_full(n * n);
            MPI_Gatherv(C_local.data(), local_rows * n, MPI_DOUBLE,
                       C_full.data(), send_counts.data(), displs.data(),
                       MPI_DOUBLE, 0, comm);
            
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    C(i, j) = C_full[i * n + j];
                }
            }
        } else {
            MPI_Gatherv(C_local.data(), local_rows * n, MPI_DOUBLE,
                       nullptr, send_counts.data(), displs.data(),
                       MPI_DOUBLE, 0, comm);
        }
        
        double total_end = MPI_Wtime();
        if (rank == 0) {
            std::cout << "Parallel multiplication time (scatter): " << (total_end - total_start) << " seconds\n";
        }
        
        return C;
    }
};

// Функция для запуска экспериментов
void run_experiment(size_t n, int rank, const std::string& input_dir = "matrices") {
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "EXPERIMENT: n = " << n << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    // Загрузка или генерация матриц
    Matrix A(n), B(n);
    
    if (rank == 0) {
        std::string a_file = input_dir + "/A_" + std::to_string(n) + ".txt";
        std::string b_file = input_dir + "/B_" + std::to_string(n) + ".txt";
        
        std::ifstream test_a(a_file);
        std::ifstream test_b(b_file);
        
        if (!test_a.good() || !test_b.good()) {
            // Генерируем случайные матрицы
            std::cout << "Generating random matrices of size " << n << "..." << std::endl;
            A.randomize();
            B.randomize();
            
            // Сохраняем для будущих запусков
            A.save_to_file(a_file);
            B.save_to_file(b_file);
        } else {
            A.load_from_file(a_file);
            B.load_from_file(b_file);
        }
    }
    
    // Рассылаем размер (хотя он известен)
    MPI_Bcast(const_cast<size_t*>(&n), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    // Если не процесс 0, создаем матрицы нужного размера
    if (rank != 0) {
        A = Matrix(n);
        B = Matrix(n);
    }
    
    // Рассылаем матрицы
    MPI_Bcast(A.data_ptr(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data_ptr(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Последовательное умножение (только на процессе 0 для сравнения)
    double seq_time = 0.0;
    Matrix C_seq(n);
    
    if (rank == 0) {
        std::cout << "Running sequential multiplication..." << std::endl;
        auto seq_start = std::chrono::high_resolution_clock::now();
        C_seq = sequential_multiply(A, B);
        auto seq_end = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(seq_end - seq_start).count();
        std::cout << "Sequential time: " << seq_time << " seconds" << std::endl;
        std::cout << "Operations: " << 2 * n * n * n << std::endl;
        std::cout << "Sequential GFLOPS: " << (2.0 * n * n * n / seq_time) / 1e9 << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Параллельное умножение
    ParallelMatrixMultiplier multiplier;
    Matrix C_par = multiplier.multiply_scatter(A, B);
    
    // Верификация на процессе 0
    if (rank == 0) {
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double diff = std::abs(C_seq(i, j) - C_par(i, j));
                if (diff > max_diff) max_diff = diff;
            }
        }
        
        std::cout << "\nVerification:" << std::endl;
        std::cout << "  Max difference: " << std::scientific << max_diff << std::endl;
        if (max_diff < 1e-9) {
            std::cout << "  Result: CORRECT ✓" << std::endl;
        } else {
            std::cout << "  Result: INCORRECT ✗" << std::endl;
        }
        
        // Сохраняем результат
        std::string result_file = "results/C_" + std::to_string(n) + ".txt";
        C_par.save_to_file(result_file);
        std::cout << "  Result saved to: " << result_file << std::endl;
    }
}

int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "============================================================" << std::endl;
        std::cout << "MATRIX MULTIPLICATION WITH MPI" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Number of processes: " << size << std::endl;
        std::cout << "============================================================" << std::endl;
    }
    
    // Параметры эксперимента
    std::vector<size_t> matrix_sizes = {200, 400, 800, 1200, 1600, 2000};
    
    // Для быстрого теста можно использовать меньшие размеры
    // std::vector<size_t> matrix_sizes = {200, 400, 800};
    
    // Создаём директорию для результатов
    if (rank == 0) {
        system("mkdir -p results");
        system("mkdir -p matrices");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Запуск экспериментов
    for (size_t n : matrix_sizes) {
        // Ограничение на размер для больших процессов
        if (n * size > 5000 && n > 800) {
            if (rank == 0) {
                std::cout << "Skipping n=" << n << " (too large for memory with " << size << " processes)" << std::endl;
            }
            continue;
        }
        
        double exp_start = MPI_Wtime();
        run_experiment(n, rank);
        double exp_end = MPI_Wtime();
        
        if (rank == 0) {
            std::cout << "Total experiment time: " << (exp_end - exp_start) << " seconds" << std::endl;
        }
    }
    
    // Вывод итоговой таблицы
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "EXPERIMENTS COMPLETED" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}