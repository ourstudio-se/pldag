import numpy as np
from pldag import PLDAG

def test_create_model():
    model = PLDAG()
    assert model is not None

def test_propagate():
    # model = PLDAG()
    # model.set_primitives("xyz")
    # model.set_composite("A", "xyz", -1)
    # model.propagate()
    # assert np.array_equal(model.get("A"), np.array([complex(1j)]))

    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("C", "x", -1)
    model.set_composite("B", "yz", -1, True)
    model.set_composite("A", "BC", -2)
    model.propagate()
    assert np.array_equal(model.get("C"), np.array([1j]))
    assert np.array_equal(model.get("B"), np.array([1j]))
    assert np.array_equal(model.get("A"), np.array([1j]))
    model.set_primitive("x", 1+1j)
    model.set_primitive("y", 0j)
    model.propagate()
    assert np.array_equal(model.get("C"), np.array([1+1j]))
    assert np.array_equal(model.get("B"), np.array([1+1j]))
    assert np.array_equal(model.get("A"), np.array([1+1j]))
    
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("C", "x", 0, True)
    model.set_composite("B", "xy", -1)
    model.set_composite("A", "BC", -2)
    model.propagate()
    assert np.array_equal(model.bounds, np.array([1j, 1j, 1j, 0j, 1j, -1-1j, 1j, -2-2j, 1j]))
    model.set_primitive("x", 1+1j)
    model.propagate()
    assert np.array_equal(model.get("C"), np.array([0j]))
    assert np.array_equal(model.get("A"), np.array([0j]))

def test_integer_bounds():

    model = PLDAG()
    model.set_primitives("z", complex(0, 10000))
    model.set_composite("A", "z", -3000)
    model.propagate()
    assert np.array_equal(model.get("z"), np.array([complex(0, 10000)]))
    assert np.array_equal(model.get("A"), np.array([complex(0, 1)]))
    model.set_primitives("z", complex(3000, 10000))
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([complex(1, 1)]))


def test_replace_composite():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("A", "xyz", -1)
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1j]))
    model.set_composite("A", "xyz", 0)
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1+1j]))

def test_get():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("A", "xyz", -1)
    for alias, expected in zip(["x", "y", "z", "A"], np.array([[1j], [1j], [1j], [1j]])):
        assert np.array_equal(model.get(alias), expected)

def test_test():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("A", "xyz", -1)

    result = model.test({"x": 1+1j}, "A")
    assert np.array_equal(result, np.array([1+1j]))

    result = model.test({"x": 1j}, "A")
    assert np.array_equal(result, np.array([1j]))

    result = model.test({"x": 0j}, "A")
    assert np.array_equal(result, np.array([1j]))

    result = model.test({"x": 0j, "y": 0j, "z": 0j}, "A")
    assert np.array_equal(result, np.array([0j]))

def test_test_second():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_composite("C", ["x"], 0, True)
    model.set_composite("B", ["y", "z"], -2)
    model.set_composite("A", ["B", "C"], -1)
    np.array_equal(
        model.test(
            {
                "x": 1+1j, 
                "y": 1+1j, 
                "z": 1+1j
            }, 
            select='A'
        ),
        np.array([1+1j])
    )
    np.array_equal(
        model.test(
            {
                "x": 1+1j, 
                "y": 0j, 
                "z": 1+1j
            }, 
            select='A'
        ),
        np.array([0j])
    )
    np.array_equal(
        model.test(
            {
                "x": 1j, 
                "y": 0j, 
                "z": 1j
            }, 
            select='A'
        ),
        np.array([1j])
    )
def test_dependencies():
    model = PLDAG() 
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
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_composite("C", ["x"], 0, True)
    model.set_composite("B", ["y", "z"], +1, True)
    model.set_composite("A", list("xyz"), +2, True)
    assert model.negated("A") == True
    assert model.negated("B") == True
    assert model.negated("C") == True
    assert model.negated("x") == False
    assert model.negated("y") == False

    assert np.array_equal(model.get("A"), np.array([1j]))
    assert np.array_equal(model.get("B"), np.array([1j]))
    assert np.array_equal(model.get("C"), np.array([1j]))

def test_delete():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_composite("C", ["x"], 0, True)
    model.set_composite("B", ["y", "z"], +1, True)
    model.set_composite("A", ["C", "B"], +2, True)
    model.delete("x")
    model.propagate()
    # Since x is removed, C should be able to be true for ever
    # Same as C = 0 >= 0
    np.array_equal(model.get("C"), np.array([1+1j]))
    # A and B should stay the same
    np.array_equal(model.get("B"), np.array([1j]))
    np.array_equal(model.get("A"), np.array([1+1j]))
    model.delete("y")
    model.propagate()

def test_cycle_detection():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_composite("A", "xyz", -1)
    # There is no way using the set functions to create a cycle
    # So we need to modify the prime table
    # Here we set that "x" as "A" as input
    model._amat[0,-1] = 1
    # Instead of an error, the propagation still works but stops.
    # It means that we have run A to times
    # Since x is a primitive boolean variable it should have 0-1 as init
    # bountds
    model.get("x")[0] == 1j
    model.propagate()
    # However, since we connected A to x, x is considered by the model
    # as a composite and will therefore eventually be evaluated as x = A >= 0
    # which is 1-1
    model.get("x")[0] == 1+1j