import pytest

from kineticslab.linalg import solve, matvec


def test_solve_identity():
    a = [[1.0, 0.0], [0.0, 1.0]]
    assert solve(a, [3.0, -4.0]) == [3.0, -4.0]


def test_solve_known_3x3():
    a = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
    b = [8.0, -11.0, -3.0]
    x = solve(a, b)
    exact = [2.0, 3.0, -1.0]
    assert max(abs(x[i] - exact[i]) for i in range(3)) < 1e-12


def test_solve_requires_pivoting():
    a = [[0.0, 1.0], [1.0, 0.0]]
    x = solve(a, [2.0, 5.0])
    assert x == [5.0, 2.0]


def test_solve_singular_raises():
    a = [[1.0, 2.0], [2.0, 4.0]]
    with pytest.raises(ZeroDivisionError):
        solve(a, [1.0, 2.0])


def test_solve_does_not_mutate_inputs():
    a = [[2.0, 1.0], [1.0, 3.0]]
    b = [1.0, 2.0]
    a0 = [row[:] for row in a]
    b0 = b[:]
    solve(a, b)
    assert a == a0 and b == b0


def test_matvec():
    a = [[1.0, 2.0], [3.0, 4.0]]
    assert matvec(a, [1.0, 1.0]) == [3.0, 7.0]
