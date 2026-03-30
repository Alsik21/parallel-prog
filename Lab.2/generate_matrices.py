import numpy as np
import sys

def save_matrix(filename, matrix):
    """Сохранение матрицы в файл как целые числа"""
    with open(filename, 'w') as f:
        f.write(f"{matrix.shape[0]} {matrix.shape[1]}\n")
        for row in matrix:
            # Принудительно преобразуем в int и убираем .0
            f.write(' '.join(str(int(x)) for x in row) + '\n')

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_matrices.py <size>")
        print("Example: python generate_matrices.py 5")
        sys.exit(1)
    
    size = int(sys.argv[1])
    
    # Генерируем целые числа от 0 до 9
    np.random.seed(42)
    A = np.random.randint(0, 10, (size, size))
    B = np.random.randint(0, 10, (size, size))
    
    # Сохраняем
    save_matrix("matrixA.txt", A)
    save_matrix("matrixB.txt", B)
    
    print(f"Generated {size}x{size} integer matrices")
    print("\nMatrix A (first 5x5 if larger):")
    print(A[:5, :5])
    print("\nMatrix B (first 5x5 if larger):")
    print(B[:5, :5])
    
    # Проверяем, что записалось
    print("\nFirst few lines of matrixA.txt:")
    with open("matrixA.txt", 'r') as f:
        for i, line in enumerate(f):
            if i < 4:  # покажем первые 4 строки
                print(line.strip())

if __name__ == "__main__":
    main()