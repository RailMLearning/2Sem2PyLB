import os
import numpy as np

import main as m


def test_create_vector():
    v = m.create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix():
    matrix = m.create_matrix()
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (5, 5)
    assert np.all((matrix >= 0) & (matrix < 1))


def test_reshape_vector():
    v = np.arange(10)
    reshaped = m.reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_vector_add():
    assert np.array_equal(
        m.vector_add(np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.array([5, 7, 9]),
    )
    assert np.array_equal(
        m.vector_add(np.array([0, 1]), np.array([1, 1])),
        np.array([1, 2]),
    )


def test_scalar_multiply():
    assert np.array_equal(
        m.scalar_multiply(np.array([1, 2, 3]), 2),
        np.array([2, 4, 6]),
    )


def test_elementwise_multiply():
    assert np.array_equal(
        m.elementwise_multiply(np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.array([4, 10, 18]),
    )


def test_dot_product():
    assert m.dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])) == 32
    assert m.dot_product(np.array([2, 0]), np.array([3, 5])) == 6


def test_matrix_multiply():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 0], [1, 2]])
    assert np.array_equal(m.matrix_multiply(a, b), a @ b)


def test_matrix_determinant():
    a = np.array([[1, 2], [3, 4]])
    assert round(m.matrix_determinant(a), 5) == -2.0


def test_matrix_inverse():
    a = np.array([[1, 2], [3, 4]])
    inv_a = m.matrix_inverse(a)
    assert np.allclose(a @ inv_a, np.eye(2))


def test_solve_linear_system():
    a = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    x = m.solve_linear_system(a, b)
    assert np.allclose(a @ x, b)


def test_load_dataset():
    # Для теста создадим временный файл
    test_data = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w") as f:
        f.write(test_data)
    try:
        data = m.load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0], [78, 81, 90])
    finally:
        os.remove("test_data.csv")


def test_statistical_analysis():
    data = np.array([10, 20, 30])
    result = m.statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_normalization():
    data = np.array([0, 5, 10])
    norm = m.normalize_data(data)
    assert np.allclose(norm, np.array([0, 0.5, 1]))


def test_plot_histogram():
    # Просто проверяем, что функция не падает
    data = np.array([1, 2, 3, 4, 5])
    m.plot_histogram(data)


def test_plot_heatmap():
    matrix = np.array([[1, 0.5], [0.5, 1]])
    m.plot_heatmap(matrix)


def test_plot_line():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    m.plot_line(x, y)


if __name__ == "__main__":
    print(
        "Запустите python -m pytest test_main.py -v "
        "для проверки лабораторной работы."
    )
