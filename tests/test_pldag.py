import numpy as np
from pldag import PLDAG

def test_create_model():
    model = PLDAG()
    assert model is not None

def test_propagate():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("A", "xyz", -1)
    assert np.array_equal(model.propagate(), np.array([[0, 1], [0, 1], [0, 1], [-1, -1], [0, 1]]))

def test_get():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("A", "xyz", -1)
    for alias, expected in zip(["x", "y", "z", "A"], np.array([[0, 1], [0, 1], [0, 1], [0, 1]])):
        assert np.array_equal(model.get(alias), expected)

def test_test():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("A", "xyz", -1)

    result = model.test({"x": (1,1)}, "A")
    assert np.array_equal(result, np.array([[1, 1]]))

    result = model.test({"x": (0,1)}, "A")
    assert np.array_equal(result, np.array([[0, 1]]))

    result = model.test({"x": (0,0)}, "A")
    assert np.array_equal(result, np.array([[0, 1]]))

    result = model.test({"x": (0,0), "y": (0,0), "z": (0,0)}, "A")
    assert np.array_equal(result, np.array([[0, 0]]))

def test_counting_properties():
    for i in range(1, 10):
        model = PLDAG(i)
        assert model.n_max == model.PRIME_HEIGHT ** i
        assert model.n_left == model.PRIME_HEIGHT ** i

def test_test_second():
    model = PLDAG(10) 
    model.set_primitives("xyz")
    model.set_composite("C", ["x"], 0, True)
    model.set_composite("B", ["y", "z"], -2)
    model.set_composite("A", ["B", "C"], -1)
    np.array_equal(
        model.test(
            {
                "x": (1,1), 
                "y": (1,1), 
                "z": (1,1)
            }, 
            select='A'
        ),
        np.array([[1,1]])
    )
    np.array_equal(
        model.test(
            {
                "x": (1,1), 
                "y": (0,0), 
                "z": (1,1)
            }, 
            select='A'
        ),
        np.array([[0,0]])
    )
    np.array_equal(
        model.test(
            {
                "x": (0,1), 
                "y": (0,0), 
                "z": (0,1)
            }, 
            select='A'
        ),
        np.array([[0,1]])
    )
def test_dependencies():
    model = PLDAG(10) 
    model.set_primitives("xyz")
    model.set_composite("C", ["x"], 0, True)
    model.set_composite("B", ["y", "z"], -2)
    model.set_composite("A", ["B", "C"], -1)
    assert model.dependencies("A") == ["C", "B", "-1"]
    assert model.dependencies("B") == ["y", "z", "-2"]
    assert model.dependencies("C") == ["x", "0"]
    assert model.dependencies("x") == []
    assert model.dependencies("y") == []

def test_negated():
    model = PLDAG(10) 
    model.set_primitives("xyz")
    model.set_composite("C", ["x"], 0, True)
    model.set_composite("B", ["y", "z"], -2)
    model.set_composite("A", ["B", "C"], -1)
    assert model.negated("A") == False
    assert model.negated("B") == False
    assert model.negated("C") == True
    assert model.negated("x") == False
    assert model.negated("y") == False