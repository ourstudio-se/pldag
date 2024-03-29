import numpy as np
from pldag import PLDAG
from itertools import chain

def test_create_model():
    model = PLDAG()
    assert model is not None

def test_propagate():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("xyz", -1, alias="A")
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([complex(1j)]))

    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("x", -1, alias="C")
    model.set_composite("yz", -1, True, alias="B")
    model.set_composite("BC", -2, alias="A")
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
    model.set_composite("x", 0, True, alias="C")
    model.set_composite("xy", -1, alias="B")
    model.set_composite("BC", -2, alias="A")
    model.propagate()
    assert np.array_equal(model.bounds, np.array([1j, 1j, 1j, 0j, 1j, -1-1j, 1j, -2-2j, 1j]))
    model.set_primitive("x", 1+1j)
    model.propagate()
    assert np.array_equal(model.get("C"), np.array([0j]))
    assert np.array_equal(model.get("A"), np.array([0j]))

def test_integer_bounds():

    model = PLDAG()
    model.set_primitives("z", complex(0, 10000))
    model.set_composite("z", -3000, alias="A")
    model.propagate()
    assert np.array_equal(model.get("z"), np.array([complex(0, 10000)]))
    assert np.array_equal(model.get("A"), np.array([complex(0, 1)]))
    model.set_primitives("z", complex(3000, 10000))
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([complex(1, 1)]))


def test_replace_composite():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("xyz", -1, alias="A")
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1j]))
    model.set_composite("xyz", 0, alias="A")
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1+1j]))

def test_get():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("xyz", -1, alias="A")
    for alias, expected in zip(["x", "y", "z", "A"], np.array([[1j], [1j], [1j], [1j]])):
        assert np.array_equal(model.get(alias), expected)

def test_test():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("xyz", -1, alias="A")

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
    model.set_composite(["x"], 0, True, alias="C")
    model.set_composite(["y", "z"], -2, alias="B")
    model.set_composite(["B", "C"], -1, alias="A")
    assert np.array_equal(
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
    # So that the model wasn't changed
    assert model.get("x") == +1j
    assert model.get("y") == +1j
    assert model.get("z") == +1j
    assert np.array_equal(
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
    assert np.array_equal(
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
    model.set_composite(["x"], 0, True, alias="C")
    model.set_composite(["y", "z"], -2, alias="B")
    model.set_composite(["B", "C"], -1, alias="A")
    assert list(chain(*model.dependencies("A").values())) == ["C", "B", "-1"]
    assert list(chain(*model.dependencies("B").values())) == ["y", "z", "-2"]
    assert list(chain(*model.dependencies("C").values())) == ["x", "0"]
    assert model.dependencies("x") == {}
    assert model.dependencies("y") == {}

def test_negated():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_composite(["x"], 0, True, alias="C")
    model.set_composite(["y", "z"], +1, True, alias="B")
    model.set_composite(list("xyz"), +2, True, alias="A")
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
    model.set_composite(["x"], 0, True, alias="C")
    model.set_composite(["y", "z"], +1, True, alias="B")
    model.set_composite(["C", "B"], +2, True, alias="A")
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
    model.set_composite("xyz", -1, alias="A")
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

def test_multiple_alias():

    model = PLDAG()
    model.set_primitives("xyz")
    model.set_composite("xyz", -1, alias="A")
    model.set_composite("xyz", -1, alias="B")
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1j]))
    assert np.array_equal(model.get("B"), np.array([1j]))

    model.set_composite(["A", "B"], -1, alias="C")
    dependencies = model.dependencies("C")
    assert dependencies[model._amap["A"]] == dependencies[model._amap["B"]]