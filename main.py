from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # безопасный backend без GUI (Tk недоступен)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def create_vector() -> np.ndarray:
    """Создать одномерный вектор со значениями от 0 до 9.

    Returns:
        np.ndarray: Целочисленный вектор формы ``(10,)`` со значениями
        ``[0, 1, ..., 9]``.
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """Создать случайную матрицу размера 5x5.

    Returns:
        np.ndarray: Матрица формы ``(5, 5)`` со случайными значениями
        с плавающей точкой в полуинтервале ``[0, 1)``.
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """Преобразовать вектор формы ``(10,)`` в матрицу формы ``(2, 5)``.

    Args:
        vec (np.ndarray): Одномерный массив, содержащий ровно 10 элементов.

    Returns:
        np.ndarray: Представление или копия массива с формой ``(2, 5)``.
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """Транспонировать матрицу.

    Args:
        mat (np.ndarray): Входная матрица корректной двумерной формы.

    Returns:
        np.ndarray: Транспонированная матрица с переставленными строками
        и столбцами.
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Сложить два вектора поэлементно.

    Args:
        a (np.ndarray): Первый входной вектор.
        b (np.ndarray): Второй входной вектор той же формы, что и ``a``.

    Returns:
        np.ndarray: Результат поэлементного сложения входных векторов.
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: float | int) -> np.ndarray:
    """Умножить вектор на скаляр.

    Args:
        vec (np.ndarray): Входной вектор или массив.
        scalar (float | int): Числовой множитель.

    Returns:
        np.ndarray: Массив, где каждый элемент умножен на ``scalar``.
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Умножить два массива поэлементно.

    Args:
        a (np.ndarray): Первый входной массив.
        b (np.ndarray): Второй входной массив совместимой формы.

    Returns:
        np.ndarray: Результат поэлементного произведения входных массивов.
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Вычислить скалярное произведение двух векторов.

    Args:
        a (np.ndarray): Первый вектор.
        b (np.ndarray): Второй вектор той же длины, что и ``a``.

    Returns:
        float: Значение скалярного произведения.
    """
    return float(np.dot(a, b))


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Умножить две матрицы по правилам матричного произведения.

    Args:
        a (np.ndarray): Левая матрица-операнд.
        b (np.ndarray): Правая матрица-операнд совместимой размерности.

    Returns:
        np.ndarray: Результат матричного произведения ``a @ b``.
    """
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """Вычислить определитель квадратной матрицы.

    Args:
        a (np.ndarray): Квадратная матрица.

    Returns:
        float: Определитель матрицы ``a``.
    """
    return float(np.linalg.det(a))


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """Вычислить обратную матрицу.

    Args:
        a (np.ndarray): Квадратная невырожденная матрица.

    Returns:
        np.ndarray: Обратная матрица.
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решить систему линейных уравнений ``Ax = b``.

    Args:
        a (np.ndarray): Матрица коэффициентов ``A``.
        b (np.ndarray): Вектор или матрица правой части ``b``.

    Returns:
        np.ndarray: Решение ``x``, удовлетворяющее уравнению ``Ax = b``.
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """Загрузить табличные данные из CSV-файла в массив NumPy.

    Args:
        path (str): Путь к CSV-файлу. По умолчанию используется
            датасет лабораторной работы.

    Returns:
        np.ndarray: Значения из CSV-файла без названий столбцов.
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict[str, float]:
    """Вычислить основные описательные статистики для одномерных данных.

    Функция возвращает среднее, медиану, стандартное отклонение, минимум,
    максимум, а также 25-й и 75-й перцентили.

    Args:
        data (np.ndarray): Одномерный числовой массив.

    Returns:
        dict[str, float]: Словарь с ключами ``mean``, ``median``, ``std``,
        ``min``, ``max``, ``percentile_25`` и ``percentile_75``.
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "percentile_25": float(np.percentile(data, 25)),
        "percentile_75": float(np.percentile(data, 75)),
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Нормализовать числовые данные в диапазон ``[0, 1]`` методом Min-Max.

    Формула:
        ``(x - min(x)) / (max(x) - min(x))``

    Args:
        data (np.ndarray): Входной числовой массив.

    Returns:
        np.ndarray: Нормализованные данные в диапазоне ``[0, 1]``.
        Если все значения одинаковые, возвращается массив нулей
        типа ``float``.
    """
    min_value = np.min(data)
    max_value = np.max(data)
    if max_value == min_value:
        return np.zeros_like(data, dtype=float)
    return (data - min_value) / (max_value - min_value)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray) -> None:
    """Построить и сохранить гистограмму распределения оценок.

    Изображение сохраняется в ``plots/histogram.png``.

    Args:
        data (np.ndarray): Одномерные числовые данные для построения
            гистограммы.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=10, edgecolor="black")
    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценка")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig("plots/histogram.png")
    plt.close()


def plot_heatmap(matrix: np.ndarray) -> None:
    """Построить и сохранить тепловую карту корреляционной матрицы.

    Изображение сохраняется в ``plots/heatmap.png``.

    Args:
        matrix (np.ndarray): Двумерная матрица со значениями корреляции.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="viridis")
    plt.title("Тепловая карта корреляции предметов")
    plt.tight_layout()
    plt.savefig("plots/heatmap.png")
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """Построить и сохранить линейный график зависимости оценки от студента.

    Изображение сохраняется в ``plots/line.png``.

    Args:
        x (np.ndarray): Индексы или идентификаторы студентов.
        y (np.ndarray): Числовые оценки, соответствующие ``x``.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title("Зависимость: студент - оценка по математике")
    plt.xlabel("Студент")
    plt.ylabel("Оценка по математике")
    plt.tight_layout()
    plt.savefig("plots/line.png")
    plt.close()
