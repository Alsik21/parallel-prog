import numpy as np
import time
import sys

def read_matrix(filename):
    """Чтение матрицы из файла (поддерживает int и float)"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Первая строка - размерность
    rows, cols = map(int, lines[0].split())
    matrix = []
    
    # Читаем данные
    for i in range(1, rows + 1):
        # Пробуем сначала как int, если не получается - как float
        try:
            row = list(map(int, lines[i].split()))
        except ValueError:
            row = list(map(float, lines[i].split()))
        matrix.append(row)
    
    return np.array(matrix)

def write_execution_time(cpp_time, python_time, matches):
    """Запись времени выполнения обеих программ"""
    with open('execution_time.txt', 'w') as f:
        f.write(f"C++ time: {cpp_time:.3f} ms\n")
        f.write(f"Python time: {python_time:.3f} ms\n")
        f.write(f"Results match: {matches}\n")
        
        if python_time > 0:
            ratio = cpp_time / python_time
            if ratio > 1:
                f.write(f"Python is {ratio:.2f}x faster\n")
            else:
                f.write(f"C++ is {1/ratio:.2f}x faster\n")

def main():
    fileA = "matrixA.txt"
    fileB = "matrixB.txt"
    fileC = "result.txt"
    
    print("\n=== VERIFICATION ===")
    
    # Читаем матрицы
    print("Reading matrices...")
    A = read_matrix(fileA)
    B = read_matrix(fileB)
    
    print(f"Matrix A shape: {A.shape}, type: {A.dtype}")
    print(f"Matrix B shape: {B.shape}, type: {B.dtype}")
    
    # Читаем время C++
    try:
        with open('execution_time.txt', 'r') as f:
            first_line = f.readline()
            cpp_time = float(first_line.split(':')[1].strip().split()[0])
    except:
        print("Could not read C++ time, using 0")
        cpp_time = 0
    
    # Умножение NumPy
    print("\nMultiplying with NumPy...")
    start_time = time.time()
    C_numpy = np.dot(A, B)
    python_time = (time.time() - start_time) * 1000
    
    # Читаем результат C++
    print("Reading C++ result...")
    C_cpp = read_matrix(fileC)
    
    # Сравниваем
    print("\n--- COMPARISON ---")
    print(f"Matrix size: {A.shape[0]} x {A.shape[1]}")
    
    # Для целых чисел проверяем точное совпадение
    if A.dtype == int and B.dtype == int:
        matches = np.array_equal(C_cpp, C_numpy)
        if matches:
            print("✅ RESULTS MATCH EXACTLY")
            max_diff = 0
        else:
            print("❌ RESULTS DO NOT MATCH")
            diff = C_cpp - C_numpy
            max_diff = np.max(np.abs(diff))
            print(f"Maximum difference: {max_diff}")
            
            # Показываем первые несколько отличий
            diff_positions = np.where(diff != 0)
            print(f"Number of different elements: {len(diff_positions[0])}")
    else:
        # Для float используем допуск
        matches = np.allclose(C_cpp, C_numpy, rtol=1e-10, atol=1e-10)
        if matches:
            print("✅ RESULTS MATCH (within tolerance)")
            max_diff = np.max(np.abs(C_cpp - C_numpy))
            print(f"Max difference: {max_diff:.2e}")
        else:
            print("❌ RESULTS DO NOT MATCH")
            max_diff = np.max(np.abs(C_cpp - C_numpy))
            print(f"Max difference: {max_diff:.2e}")
    
    # Записываем время
    write_execution_time(cpp_time, python_time, matches)
    
    print(f"\nC++ time: {cpp_time:.3f} ms")
    print(f"Python time: {python_time:.3f} ms")
    
    if python_time > 0:
        ratio = cpp_time / python_time
        if ratio > 1:
            print(f"Python is {ratio:.2f}x faster")
        else:
            print(f"C++ is {1/ratio:.2f}x faster")

if __name__ == "__main__":
    main()