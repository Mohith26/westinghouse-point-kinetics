"""Tiny dense linear algebra helpers, pure Python.

The kinetics system is only 7x7 so a straightforward Gaussian elimination
with partial pivoting is plenty. No numpy on purpose: the solver itself is
meant to be dependency free, numpy/scipy are reserved for the validation
oracle.
"""


def solve(a, b):
    """Solve A x = b for dense square A (list of lists), b a list.

    Gaussian elimination with partial pivoting. A and b are not modified.
    """
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) == 0.0:
            raise ZeroDivisionError("singular matrix in solve()")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] * inv
            if f != 0.0:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n]
        for c in range(r + 1, n):
            s -= m[r][c] * x[c]
        x[r] = s / m[r][r]
    return x


def matvec(a, v):
    """Dense matrix-vector product."""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]
