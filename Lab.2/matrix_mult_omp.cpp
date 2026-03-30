#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <cstdlib>

using namespace std;
using namespace chrono;

// Функция для чтения матрицы из файла
vector<vector<int>> readMatrix(const string& filename, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        exit(1);
    }
    
    file >> rows >> cols;
    vector<vector<int>> matrix(rows, vector<int>(cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file >> matrix[i][j];
        }
    }
    
    file.close();
    return matrix;
}

// Функция для записи матрицы в файл
void writeMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось создать файл " << filename << endl;
        exit(1);
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    file << rows << " " << cols << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << matrix[i][j];
            if (j < cols - 1) file << " ";
        }
        file << endl;
    }
    
    file.close();
}

// Параллельное умножение матриц с использованием OpenMP
vector<vector<int>> multiplyMatricesOMP(const vector<vector<int>>& A, 
                                         const vector<vector<int>>& B, 
                                         int num_threads) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    
    // Проверка совместимости размеров
    if ((int)A[0].size() != p) {
        cerr << "Ошибка: несовместимые размеры матриц" << endl;
        exit(1);
    }
    
    vector<vector<int>> C(n, vector<int>(m, 0));
    
    // Устанавливаем количество потоков
    omp_set_num_threads(num_threads);
    
    double start_time = omp_get_wtime();
    
    // Параллельное умножение
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int sum = 0;
            for (int k = 0; k < p; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    double end_time = omp_get_wtime();
    cout << "OpenMP время вычислений: " << (end_time - start_time) * 1000 << " мс" << endl;
    
    return C;
}

// Последовательное умножение для сравнения
vector<vector<int>> multiplyMatricesSequential(const vector<vector<int>>& A, 
                                                const vector<vector<int>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    
    vector<vector<int>> C(n, vector<int>(m, 0));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int sum = 0;
            for (int k = 0; k < p; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    return C;
}

// Сравнение матриц
bool compareMatrices(const vector<vector<int>>& C1, const vector<vector<int>>& C2) {
    if (C1.size() != C2.size() || C1[0].size() != C2[0].size()) {
        return false;
    }
    
    for (size_t i = 0; i < C1.size(); i++) {
        for (size_t j = 0; j < C1[0].size(); j++) {
            if (C1[i][j] != C2[i][j]) {
                cout << "Различие в позиции [" << i << "][" << j << "]: "
                     << C1[i][j] << " vs " << C2[i][j] << endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Проверка аргументов
    if (argc != 2) {
        cerr << "Использование: " << argv[0] << " <количество_потоков>" << endl;
        cerr << "Пример: " << argv[0] << " 4" << endl;
        return 1;
    }
    
    int num_threads = atoi(argv[1]);
    if (num_threads <= 0) {
        cerr << "Ошибка: количество потоков должно быть положительным" << endl;
        return 1;
    }
    
    cout << "\n=== УМНОЖЕНИЕ МАТРИЦ С OPENMP ===" << endl;
    cout << "Количество потоков: " << num_threads << endl;
    
    // Проверка поддержки OpenMP
    #ifdef _OPENMP
        cout << "OpenMP поддерживается" << endl;
    #else
        cout << "OpenMP НЕ поддерживается!" << endl;
    #endif
    
    // Чтение матриц
    int rowsA, colsA, rowsB, colsB;
    
    cout << "\nЧтение матрицы A из файла matrixA.txt..." << endl;
    vector<vector<int>> A = readMatrix("matrixA.txt", rowsA, colsA);
    
    cout << "Чтение матрицы B из файла matrixB.txt..." << endl;
    vector<vector<int>> B = readMatrix("matrixB.txt", rowsB, colsB);
    
    cout << "Размер матрицы A: " << rowsA << " x " << colsA << endl;
    cout << "Размер матрицы B: " << rowsB << " x " << colsB << endl;
    
    // Проверка возможности умножения
    if (colsA != rowsB) {
        cerr << "\nОшибка: невозможно умножить матрицы!" << endl;
        cerr << "Количество столбцов A (" << colsA << ") != количеству строк B (" << rowsB << ")" << endl;
        return 1;
    }
    
    // Параллельное умножение
    cout << "\nВыполняется параллельное умножение..." << endl;
    auto start_parallel = high_resolution_clock::now();
    vector<vector<int>> C_parallel = multiplyMatricesOMP(A, B, num_threads);
    auto end_parallel = high_resolution_clock::now();
    auto parallel_time = duration_cast<milliseconds>(end_parallel - start_parallel).count();
    
    cout << "Общее время параллельного умножения: " << parallel_time << " мс" << endl;
    
    // Для небольших матриц выполняем проверку
    if (rowsA <= 500) {
        cout << "\nВыполняется последовательное умножение для проверки..." << endl;
        auto start_seq = high_resolution_clock::now();
        vector<vector<int>> C_seq = multiplyMatricesSequential(A, B);
        auto end_seq = high_resolution_clock::now();
        auto sequential_time = duration_cast<milliseconds>(end_seq - start_seq).count();
        
        cout << "Время последовательного умножения: " << sequential_time << " мс" << endl;
        
        // Сравнение результатов
        cout << "\n--- ПРОВЕРКА РЕЗУЛЬТАТОВ ---" << endl;
        if (compareMatrices(C_parallel, C_seq)) {
            cout << "✅ РЕЗУЛЬТАТЫ СОВПАДАЮТ" << endl;
            double speedup = (double)sequential_time / parallel_time;
            cout << "Ускорение: " << fixed << setprecision(2) << speedup << "x" << endl;
        } else {
            cout << "❌ РЕЗУЛЬТАТЫ НЕ СОВПАДАЮТ!" << endl;
        }
    }
    
    // Сохранение результата
    writeMatrix("result.txt", C_parallel);
    cout << "\nРезультат сохранен в файл result.txt" << endl;
    
    // Сохранение времени выполнения
    ofstream time_file("execution_time.txt");
    if (time_file.is_open()) {
        time_file << "C++ OpenMP time: " << parallel_time << " ms" << endl;
        time_file << "Threads: " << num_threads << endl;
        time_file.close();
    }
    
    // Информация о системе
    cout << "\n--- ИНФОРМАЦИЯ О СИСТЕМЕ ---" << endl;
    cout << "Доступно ядер: " << omp_get_num_procs() << endl;
    cout << "Максимальное количество потоков: " << omp_get_max_threads() << endl;
    
    return 0;
}