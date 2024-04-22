import numpy as np
from pldag import PLDAG
from hypothesis import given, strategies, assume, settings

def composite_proposition_strategy():
    """
        Returns a strategy of all necessary components of a composite proposition.
        First element is the children, second bias and third if negated or not.
    """
    return strategies.tuples(
        strategies.sets(strategies.text(), min_size=1, max_size=5),
        strategies.integers(min_value=1, max_value=5),
        strategies.booleans(),
    )

def composite_proposition_strategies():
    return strategies.lists(composite_proposition_strategy(), min_size=1, max_size=10)

@given(composite_proposition_strategies())
def model_strategy(composites):
    model = PLDAG()
    for children, bias, negated in composites:
        model.set_primitives(children)
        model.set_atleast(children, bias, negated)
    return model

@given(composite_proposition_strategy(), composite_proposition_strategy())
def test_id_generation(comp1, comp2):
    # Check if the comp1 and comp2 are the same,
    # the id should be the same.
    # (comp1 == comp2) <=> (id(comp1) == id(comp2))
    assert (comp1 == comp2) == (PLDAG._composite_id(*comp1) == PLDAG._composite_id(*comp2))

def test_create_model():
    model = PLDAG()
    assert model is not None

def test_propagate():
    model = PLDAG()
    model.set_primitives("xyz")
    a=model.set_atleast("xyz", 1)
    model.propagate_downstream()
    assert np.array_equal(model.get(a), np.array([complex(1j)]))

    model = PLDAG()
    model.set_primitives("xyz")
    c = model.set_atleast("x", 1)
    b = model.set_atmost("yz", 1)
    a = model.set_atleast([b,c], 2)
    model.propagate_downstream()
    assert np.array_equal(model.get(c), np.array([1j]))
    assert np.array_equal(model.get(b), np.array([1j]))
    assert np.array_equal(model.get(a), np.array([1j]))
    model.set_primitive("x", 1+1j)
    model.set_primitive("y", 0j)
    model.propagate_downstream()
    assert np.array_equal(model.get(c), np.array([1+1j]))
    assert np.array_equal(model.get(b), np.array([1+1j]))
    assert np.array_equal(model.get(a), np.array([1+1j]))
    
    model = PLDAG()
    model.set_primitives("xyz")
    c=model.set_atmost("x", 0)
    b=model.set_atleast("xy", 1)
    a=model.set_atleast([b, c], 2)
    model.propagate_downstream()
    assert np.array_equal(model.bounds, np.array([1j, 1j, 1j, 1j, 1j, 1j]))
    model.set_primitive("x", 1+1j)
    model.propagate_downstream()
    assert np.array_equal(model.get(c), np.array([0j]))
    assert np.array_equal(model.get(a), np.array([0j]))

def test_integer_bounds():

    model = PLDAG()
    model.set_primitives("z", complex(0, 10000))
    a=model.set_atleast("z", 3000)
    model.propagate_downstream()
    assert np.array_equal(model.get("z"), np.array([complex(0, 10000)]))
    assert np.array_equal(model.get(a), np.array([complex(0, 1)]))
    model.set_primitives("z", complex(3000, 10000))
    model.propagate_downstream()
    assert np.array_equal(model.get(a), np.array([complex(1, 1)]))


def test_replace_composite():
    model = PLDAG()
    model.set_primitives("xyz")
    a = model.set_atleast("xyz", 1)
    model.propagate_downstream()
    assert np.array_equal(model.get(a), np.array([1j]))
    a = model.set_atleast("xyz", 0)
    model.propagate_downstream()
    assert np.array_equal(model.get(a), np.array([1+1j]))

def test_get():
    model = PLDAG()
    model.set_primitives("xyz")
    a=model.set_atleast("xyz", -1)
    for id, expected in zip(["x", "y", "z", a], np.array([[1j], [1j], [1j], [1j]])):
        assert np.array_equal(model.get(id), expected)

def test_test():
    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_atleast("xyz", 1)

    assert model.test({"x": 1+1j}).get(id) == 1+1j
    assert model.test({"x": 1j}).get(id) == 1j
    assert model.test({"x": 0j}).get(id) == 1j
    assert model.test({"x": 0j, "y": 0j, "z": 0j}).get(id) == 0j

    model = PLDAG()
    model.set_primitives("xyz")
    a = model.set_atmost(["x","y"], 1)
    assert model.test({"x": 1+1j}).get(a) == 1j
    assert model.test({"x": 1+1j, "y": 0j}).get(a) == 1+1j
    assert model.test({"x": 1+1j, "y": 1+1j}).get(a) == 0j

    model = PLDAG()
    model.set_primitives("xy")
    a = model.set_and(["x"])
    b = model.set_not(["y"])
    c = model.set_and([a, b])
    assert model.test({"x": 1+1j, "y": 0j}).get(c) == 1+1j
    assert model.test({"y": 0j, "x": 1+1j}).get(c) == 1+1j

def test_test_second():
    model = PLDAG() 
    model.set_primitives("xyz")
    c = model.set_atmost(["x"], 0)
    b = model.set_atleast(["y", "z"], 2)
    a = model.set_atleast([b, c], 1)
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
    c=model.set_atmost(["x"], 0)
    b=model.set_atleast(["y", "z"], 2)
    a=model.set_atleast([b,c], 1)
    assert model.dependencies(a) == {c,b}
    assert model.dependencies(b) == {"y", "z"}
    assert model.dependencies(c) == {"x"}

def test_negated():
    model = PLDAG() 
    model.set_primitives("xyz")
    c=model.set_atmost(["x"], 1)
    b=model.set_atmost(["y", "z"], 2)
    a=model.set_atleast("xyz", 2)
    assert model.negated(a) == False
    assert model.negated(b) == True
    assert model.negated(c) == True

    assert np.array_equal(model.get(a), np.array([1j]))
    assert np.array_equal(model.get(b), np.array([1j]))
    assert np.array_equal(model.get(c), np.array([1j]))

def test_cycle_detection():
    model = PLDAG() 
    model.set_primitives("xyz")
    model.set_atleast("xyz", -1)
    # There is no way using the set functions to create a cycle
    # So we need to modify the prime table
    # Here we set that "x" as "A" as input
    model._amat[0,-1] = 1
    # Instead of an error, the propagation still works but stops.
    # It means that we have run A to times
    # Since x is a primitive boolean variable it should have 0-1 as init
    # bountds
    model.get("x")[0] == 1j
    model.propagate_downstream()
    # However, since we connected A to x, x is considered by the model
    # as a composite and will therefore eventually be evaluated as x = A >= 0
    # which is 1-1
    model.get("x")[0] == 1+1j

def test_multiple_pointers():

    model = PLDAG()
    model.set_primitives("xyz")
    a=model.set_atleast("xyz", 1)
    b=model.set_atleast("xyz", 1)
    model.propagate_downstream()
    assert np.array_equal(model.get(a), np.array([1j]))
    assert np.array_equal(model.get(b), np.array([1j]))

    c=model.set_atleast([a, b], -1)
    dependencies = list(model.dependencies(c))
    assert len(dependencies) == 1
    assert (dependencies[0] == a) and (dependencies[0] == b)

def test_to_polyhedron():

    model = PLDAG()
    model.set_primitive("a", -5+3j)
    model.set_primitive("b", 2j)
    model.set_primitive("c", -4+4j)
    model.set_primitive("d", -4+5j)
    model.set_primitive("e", 1j)
    model.set_atleast("be", 3)
    model.set_atleast("abcd", -9)
    model.set_atmost("abcd", 5)
    A,b,vs = model.to_polyhedron()
    assert np.array_equal(A, np.array([
        [ 0,  1,  0,  0,  1, -3,  0,  0],
        [ 1,  1,  1,  1,  0,  0, -4,  0],
        [-1, -1, -1, -1,  0,  0,  0, -9],
        [ 1,  0,  0,  0,  0,  0,  0,  0],
        [-1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0, -1,  0,  0,  0,  0],
    ]))
    assert np.array_equal(b, np.array([0., -13., -14.,  -5.,  -3.,   0.,  -2.,  -4.,  -4.,  -4.,  -5.]))

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
    a=model.set_atleast("xyz", 1)
    A,b,vs = model.to_polyhedron()
    assert np.array_equal(A, np.array([[1,1,1,-1]]))
    assert np.array_equal(b, np.array([ 0]))
    assert len(vs) == 4

    A,b,vs = model.to_polyhedron(**{a: 1, "x": 1, "y": 1, "z": 1})
    assert np.array_equal(A, np.array([[1,1,1,-1], [1,1,1,1]]))
    assert np.array_equal(b, np.array([ 0, 4]))

    A,b,vs = model.to_polyhedron(**{a: -1, "x": -1, "y": -1, "z": -1})
    assert np.array_equal(A, np.array([[1,1,1,-1], [1,1,1,1]]))
    assert np.array_equal(b, np.array([ 0, -4]))

    A,b,vs = model.to_polyhedron(**{a: 2, "x": 2, "y": 0, "z": 0})
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
    model.set_primitives("xy")
    id = model.set_imply("x", "y")
    assert model.test({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.test({"x": 1j, "y": 1+1j}).get(id) == 1+1j
    assert model.test({"x": 1+1j, "y": 0j}).get(id) == 0j
    assert model.test({"x": 0j, "y": 1j}).get(id) == 1+1j
    assert model.test({"x": 0j, "y": 1+1j}).get(id) == 1+1j

    model = PLDAG()
    model.set_primitives("xy")
    id = model.set_xor("xy")
    assert model.test({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.test({"x": 1j, "y": 1+1j}).get(id) == 1j
    assert model.test({"x": 1+1j, "y": 0j}).get(id) == 1+1j
    assert model.test({"x": 0j, "y": 1j}).get(id) == 1j
    assert model.test({"x": 0j, "y": 1+1j}).get(id) == 1+1j

def test_sub():
    model = PLDAG()
    model.set_primitives(["m1","m2","c1","c2","e1","e2","e3","g1","g2"])
    sub_id = model.set_and([
        model.set_xor(["m1","m2"]),
        model.set_xor(["c1","c2"]),
        model.set_xor(["e1","e2","e3"]),
        model.set_imply("m1", model.set_or(["e1", "e2"])),
        model.set_imply("m2", model.set_or(["e1", "e3"])),
        model.set_imply(model.set_or(["e1", "e2"]), "g1"),
        model.set_imply("e3", "g2"),
    ])
    not_included = [model.set_and(["m1", "e1"]), model.set_and(["m2", "e2"])]
    sub_model = model.sub([sub_id])
    assert all(map(lambda id: id not in sub_model._imap, not_included))

def test_propagate_upstream():

    model = PLDAG()
    model.set_primitive("x", 10j)
    model.set_primitive("y")
    id1 = model.set_atleast(["x"], 10)
    id2 = model.set_atleast(["x", "y"], 11)
    model._dvec[model._imap[id1]] = 1+1j
    model._dvec[model._imap[id2]] = 1+1j
    model.get("x") == 10j
    model.get("y") == 1j
    model.propagate_upstream()
    model.get("x") == 10+10j
    model.get("y") == 1+1j
    
    model = PLDAG()
    model.set_primitive("w", 10j)
    model.set_primitives("xyz")
    top = model.set_and([
        model.set_imply(
            model.set_atleast(["w"], 5),
            "x",
            alias="(w >= 5) -> x"
        ),
        model.set_imply(
            model.set_and([
                model.set_atleast(["w"], 2),
                model.set_atmost(["w"], 4),
            ]),
            "y",
            alias="(2 <= w <= 4) -> y"
        ),
        model.set_imply(
            model.set_atmost(["w"], 1),
            "z",
            alias="(w <= 1) -> z"
        ),
        model.set_xor(["x", "y", "z"], alias="x XOR y XOR z")
    ], alias="top")
    model._dvec[model._imap["w"]] = 5+5j
    model.propagate_downstream()
    model._dvec[model._imap[top]] = 1+1j
    model.propagate_upstream()
    model.get("w") == 5+5j
    model.get("x") == 1+1j
