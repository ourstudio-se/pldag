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
    
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("C", "x", 0, True)
    model.set_composite("B", "xy", -1)
    model.set_composite("A", "BC", -2)
    assert np.array_equal(model.propagate(), np.array([[0, 1], [0, 1], [0, 1], [0, 0], [0, 1], [-1, -1], [0, 1], [-2,-2], [0, 1]]))
    model.set_primitive("x", (1,1))
    model.propagate()
    assert np.array_equal(model.get("C"), np.array([0,0]))
    assert np.array_equal(model.get("A"), np.array([0,0]))

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
    model.set_composite("B", ["y", "z"], +1, True)
    model.set_composite("A", list("xyz"), +2, True)
    assert model.negated("A") == True
    assert model.negated("B") == True
    assert model.negated("C") == True
    assert model.negated("x") == False
    assert model.negated("y") == False

    assert np.array_equal(model.get("A"), np.array([0,1]))
    assert np.array_equal(model.get("B"), np.array([0,1]))
    assert np.array_equal(model.get("C"), np.array([0,1]))

def test_delete():
    model = PLDAG(3) 
    model.set_primitives("xyz")
    model.set_composite("C", ["x"], 0, True)
    model.set_composite("B", ["y", "z"], +1, True)
    model.set_composite("A", ["C", "B"], +2, True)
    model.delete("x")
    model.propagate()
    # Since x is removed, C should be able to be true for ever
    # Same as C = 0 >= 0
    np.array_equal(model.get("C"), np.array([1,1]))
    # A and B should stay the same
    np.array_equal(model.get("B"), np.array([0,1]))
    np.array_equal(model.get("A"), np.array([1,1]))
    model.delete("y")
    model.propagate()