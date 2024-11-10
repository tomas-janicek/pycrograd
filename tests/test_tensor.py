from pycrograd import tensor


def test_create_zeroed() -> None:
    m1 = tensor.Matrix.create_zeroed(rows=3, cols=5)

    assert m1.rows == 3
    assert m1.cols == 5
    assert m1.data[0][0] == 0
    assert m1.data[2][4] == 0


def test_create_scalar() -> None:
    m1 = tensor.Matrix.create_scalar(value=32.0)

    assert m1.rows == 1
    assert m1.cols == 1
    assert m1.data[0][0] == 32.0


def test_create_vector() -> None:
    m1 = tensor.Matrix.create_vector([1.0, 2.0, 3.0])

    assert m1.rows == 3
    assert m1.cols == 1
    assert m1.data[0][0] == 1.0
    assert m1.data[2][0] == 3.0
