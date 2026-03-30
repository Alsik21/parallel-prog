import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_results(csv_file):
    """Анализ результатов экспериментов"""
    
    if not os.path.exists(csv_file):
        print(f"Файл {csv_file} не найден!")
        return
    
    df = pd.read_csv(csv_file)
    
    print("=== АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ ===\n")
    print(df.to_string())
    
    sizes = df['Size'].unique()
    threads = df['Threads'].unique()
    
    plt.figure(figsize=(12, 8))
    
    for size in sorted(sizes):
        data = df[df['Size'] == size]
        plt.plot(data['Threads'], data['Time_ms'], 'o-', label=f'{size}x{size}', linewidth=2, markersize=8)
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Время выполнения (мс)', fontsize=12)
    plt.title('Зависимость времени умножения матриц от количества потоков', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(threads)
    
    plt.savefig('results/time_vs_threads.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    
    for size in sorted(sizes):
        data = df[df['Size'] == size]
        plt.plot(data['Threads'], data['Speedup'], 'o-', label=f'{size}x{size}', linewidth=2, markersize=8)
    
    plt.plot(threads, threads, 'k--', label='Идеальное ускорение', linewidth=2, alpha=0.7)
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Ускорение', fontsize=12)
    plt.title('Ускорение при параллельном умножении матриц', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(threads)
    
    plt.savefig('results/speedup_vs_threads.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    
    for size in sorted(sizes):
        data = df[df['Size'] == size]
        efficiency = data['Speedup'] / data['Threads']
        plt.plot(data['Threads'], efficiency, 'o-', label=f'{size}x{size}', linewidth=2, markersize=8)
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Эффективность', fontsize=12)
    plt.title('Эффективность параллелизации', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(threads)
    plt.ylim(0, 1.1)
    
    plt.savefig('results/efficiency_vs_threads.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===\n")
    pivot_time = df.pivot(index='Size', columns='Threads', values='Time_ms')
    pivot_speedup = df.pivot(index='Size', columns='Threads', values='Speedup')
    
    print("Время выполнения (мс):")
    print(pivot_time)
    print("\nУскорение:")
    print(pivot_speedup)
    
    pivot_time.to_csv('results/pivot_time.csv')
    pivot_speedup.to_csv('results/pivot_speedup.csv')
    
    print("\n=== СТАТИСТИКА ===\n")
    for size in sorted(sizes):
        data = df[df['Size'] == size]
        seq_time = data[data['Threads'] == 1]['Time_ms'].values[0]
        best_thread = data.loc[data['Time_ms'].idxmin()]
        
        print(f"Размер {size}x{size}:")
        print(f"  - Последовательное время: {seq_time:.0f} мс")
        print(f"  - Лучшее время: {best_thread['Time_ms']:.0f} мс при {best_thread['Threads']:.0f} потоках")
        print(f"  - Максимальное ускорение: {best_thread['Speedup']:.2f}x")
        print()

if __name__ == "__main__":
    analyze_results('results/experiment_results.csv')