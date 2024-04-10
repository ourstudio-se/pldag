import numpy as np
from pldag import PLDAG
from itertools import chain

def test_create_model():
    model = PLDAG()
    assert model is not None

def test_propagate():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_atleast("xyz", 1, aliases=["A"])
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([complex(1j)]))

    model = PLDAG()
    model.set_primitives("xyz")
    model.set_atleast("x", 1, aliases=["C"])
    model.set_atmost("yz", 1, aliases=["B"])
    model.set_atleast("BC", 2, aliases=["A"])
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
    model.set_atmost("x", 0, aliases=["C"])
    model.set_atleast("xy", 1, aliases=["B"])
    model.set_atleast("BC", 2, aliases=["A"])
    model.propagate()
    assert np.array_equal(model.bounds, np.array([1j, 1j, 1j, 1j, 1j, 1j]))
    model.set_primitive("x", 1+1j)
    model.propagate()
    assert np.array_equal(model.get("C"), np.array([0j]))
    assert np.array_equal(model.get("A"), np.array([0j]))

def test_integer_bounds():

    model = PLDAG()
    model.set_primitives("z", complex(0, 10000))
    model.set_atleast("z", 3000, aliases=["A"])
    model.propagate()
    assert np.array_equal(model.get("z"), np.array([complex(0, 10000)]))
    assert np.array_equal(model.get("A"), np.array([complex(0, 1)]))
    model.set_primitives("z", complex(3000, 10000))
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([complex(1, 1)]))


def test_replace_composite():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_atleast("xyz", 1, aliases=["A"])
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1j]))
    model.set_atleast("xyz", 0, aliases=["A"])
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1+1j]))

def test_get():
    model = PLDAG()
    model.set_primitives("xyz")
    model.set_atleast("xyz", -1, aliases=["A"])
    for alias, expected in zip(["x", "y", "z", "A"], np.array([[1j], [1j], [1j], [1j]])):
        assert np.array_equal(model.get(alias), expected)

def test_test():
    model = PLDAG()
    id = model.set_atleast("xyz", 1, aliases=["A"])

    assert model.test({"x": 1+1j}).get(id) == 1+1j
    assert model.test({"x": 1j}).get(id) == 1j
    assert model.test({"x": 0j}).get(id) == 1j
    assert model.test({"x": 0j, "y": 0j, "z": 0j}).get(id) == 0j

    model = PLDAG()
    a = model.set_atmost(["x","y"], 1)
    assert model.test({"x": 1+1j}).get(a) == 1j
    assert model.test({"x": 1+1j, "y": 0j}).get(a) == 1+1j
    assert model.test({"x": 1+1j, "y": 1+1j}).get(a) == 0j

def test_test_second():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_atmost(["x"], 0, aliases=["C"])
    model.set_atleast(["y", "z"], 2, aliases=["B"])
    a = model.set_atleast(["B", "C"], 1, aliases=["A"])
    assert model.test({
        "x": 1+1j, 
        "y": 1+1j, 
        "z": 1+1j
    }).get(a) == 1+1j
    # So that the model wasn't changed
    assert model.get("x") == +1j
    assert model.get("y") == +1j
    assert model.get("z") == +1j
    assert model.test({
        "x": 1+1j, 
        "y": 0j, 
        "z": 1+1j
    }).get(a) == 0j
    assert model.test({
        "x": 1j, 
        "y": 0j, 
        "z": 1j
    }).get(a) == 1j

def test_dependencies():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_atmost(["x"], 0, aliases=["C"])
    model.set_atleast(["y", "z"], 2, aliases=["B"])
    model.set_atleast(["B", "C"], 1, aliases=["A"])
    assert list(chain(*model.dependencies("A").values())) == ["C", "B"]
    assert list(chain(*model.dependencies("B").values())) == ["y", "z"]
    assert list(chain(*model.dependencies("C").values())) == ["x"]

def test_negated():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_atmost(["x"], 1, aliases=["C"])
    model.set_atmost(["y", "z"], 2, aliases=["B"])
    model.set_atleast("xyz", 2, aliases=["A"])
    assert model.negated("A") == False
    assert model.negated("B") == True
    assert model.negated("C") == True

    assert np.array_equal(model.get("A"), np.array([1j]))
    assert np.array_equal(model.get("B"), np.array([1j]))
    assert np.array_equal(model.get("C"), np.array([1j]))

def test_delete():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_atmost(["x"], 0, aliases=["C"])
    model.set_atmost(["y", "z"], 1, aliases=["B"])
    model.set_atmost(["C", "B"], 2, aliases=["A"])
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
    model.set_atleast("xyz", -1, aliases=["A"])
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
    model.set_atleast("xyz", 1, aliases=["A"])
    model.set_atleast("xyz", 1, aliases=["B"])
    model.propagate()
    assert np.array_equal(model.get("A"), np.array([1j]))
    assert np.array_equal(model.get("B"), np.array([1j]))

    model.set_atleast(["A", "B"], -1, aliases=["C"])
    dependencies = model.dependencies("C")
    assert dependencies[model._amap["A"]] == dependencies[model._amap["B"]]

def test_to_polyhedron():

    model = PLDAG()
    model.set_primitive("a", -5+3j)
    model.set_primitive("b", 2j)
    model.set_primitive("c", -4+4j)
    model.set_primitive("d", -4+5j)
    model.set_primitive("e", 1j)
    model.set_atleast("be", 3)
    model.set_atleast("abcd", -9, aliases=["A"])
    model.set_atmost("abcd", 5, aliases=["B"])
    A,b,vs = model.to_polyhedron()
    assert np.array_equal(A, np.array([
        [  0,   1,   0,   0,   1,  -3,   0,   0],
        [  1,   1,   1,   1,   0,   0,  -4,   0],
        [ -1,  -1,  -1,  -1,   0,   0,   0,  -9],
    ]))
    assert np.array_equal(b, np.array([0,-13,-14]))

    model = PLDAG()
    model.set_primitives("xyz")
    model.set_atleast("xyz", 1)
    model.set_primitives("abc")
    model.set_atleast("abc", 3)
    # -(+x +a -2 >= 0) becomes -x-a +1 >= 0 (at most 1)
    model._set_gelineq("xa", -2, True)
    # -(+y +b -1 >= 0) becomes -x-a +0 >= 0 (at most 0)
    model._set_gelineq("yb", -1, True)
    A,b,vs = model.to_polyhedron()
    assert np.array_equal(
        A, 
        np.array([
            [ 1, 1, 1,-1, 0, 0, 0, 0, 0, 0], 
            [ 0, 0, 0, 0, 1, 1, 1,-3, 0, 0], 
            [-1, 0, 0, 0,-1, 0, 0, 0,-1, 0],
            [ 0,-1, 0, 0, 0,-1, 0, 0, 0,-2],
        ])
    )
    assert np.array_equal(b, np.array([ 0, 0,-2,-2]))

    model = PLDAG()
    model.set_primitives("xyz")
    model.set_atleast("xyz", 1, aliases=["A"])
    A,b,vs = model.to_polyhedron()
    assert np.array_equal(A, np.array([[1,1,1,-1]]))
    assert np.array_equal(b, np.array([ 0]))
    assert len(vs) == 4

    A,b,vs = model.to_polyhedron(fix={"A": 1, "x": 1, "y": 1, "z": 1})
    assert np.array_equal(A, np.array([[1,1,1,-1], [1,1,1,1]]))
    assert np.array_equal(b, np.array([ 0, 4]))

    A,b,vs = model.to_polyhedron(fix={"A": -1, "x": -1, "y": -1, "z": -1})
    assert np.array_equal(A, np.array([[1,1,1,-1], [1,1,1,1]]))
    assert np.array_equal(b, np.array([ 0, -4]))

    A,b,vs = model.to_polyhedron(fix={"A": 2, "x": 2, "y": 0, "z": 0})
    assert np.array_equal(A, np.array([[1,1,1,-1], [1,0,0,1], [0,1,1,0]]))
    assert np.array_equal(b, np.array([ 0, 4, 0]))

def test_logical_operators():

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_or("xyz")
    assert model.test({"x": 1j}).get(id) == 1j
    assert model.test({"x": 1+1j}).get(id) == 1+1j

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_and("xyz")
    assert model.test({"x": 1j, "y": 1+1j}).get(id) == 1j
    assert model.test({"x": 1+1j, "y": 1+1j, "z": 1+1j}).get(id) == 1+1j

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_not("xyz")
    assert model.test({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.test({"x": 1j, "y": 1+1j}).get(id) == 0j
    assert model.test({"x": 1+1j, "y": 1j}).get(id) == 0j
    assert model.test({"x": 1+1j, "y": 1+1j, "z": 1+1j}).get(id) == 0j
    assert model.test({"x": 0j, "y": 0j, "z": 0j}).get(id) == 1+1j

    model = PLDAG()
    id = model.set_imply("x", "y")
    assert model.test({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.test({"x": 1j, "y": 1+1j}).get(id) == 1+1j
    assert model.test({"x": 1+1j, "y": 0j}).get(id) == 0j
    assert model.test({"x": 0j, "y": 1j}).get(id) == 1+1j
    assert model.test({"x": 0j, "y": 1+1j}).get(id) == 1+1j

    model = PLDAG()
    id = model.set_xor("xy")
    assert model.test({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.test({"x": 1j, "y": 1+1j}).get(id) == 1j
    assert model.test({"x": 1+1j, "y": 0j}).get(id) == 1+1j
    assert model.test({"x": 0j, "y": 1j}).get(id) == 1j
    assert model.test({"x": 0j, "y": 1+1j}).get(id) == 1+1j
