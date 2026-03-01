import numpy as np
import sys

def save_matrix(filename, matrix):
    """Сохранение матрицы в файл"""
    with open(filename, 'w') as f:
        f.write(f"{matrix.shape[0]} {matrix.shape[1]}\n")
        for row in matrix:
            # Записываем как целые числа (без десятичной точки)
            f.write(' '.join(str(int(x)) for x in row) + '\n')

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_matrices.py <size>")
        print("Example: python generate_matrices.py 5")
        sys.exit(1)
    
    size = int(sys.argv[1])
    
    # Генерация целых чисел от 0 до 9
    np.random.seed(42)  # для воспроизводимости
    A = np.random.randint(0, 10, (size, size))
    B = np.random.randint(0, 10, (size, size))
    
    # Сохранение в файлы
    save_matrix("matrixA.txt", A)
    save_matrix("matrixB.txt", B)
    
    print(f"Generated integer matrices {size}x{size}")
    print("Files:")
    print("  - matrixA.txt")
    print("  - matrixB.txt")
    
    # Покажем первые несколько элементов для проверки
    print("\nFirst 3x3 of matrix A:")
    print(A[:3, :3])
    print("\nFirst 3x3 of matrix B:")
    print(B[:3, :3])

if __name__ == "__main__":
    main()