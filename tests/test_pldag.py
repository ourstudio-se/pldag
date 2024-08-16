import numpy as np
from pldag import *
from hypothesis import given, strategies

def composite_coefficient_variables():
    return strategies.tuples(
        strategies.text(),
        strategies.integers(min_value=-5, max_value=5),
    )
    
def composite_proposition_strategy():
    """
        Returns a strategy of all necessary components of a composite proposition.
        First element is the children, second bias and third if negated or not.
    """
    return strategies.tuples(
        strategies.sets(composite_coefficient_variables(), min_size=1, max_size=5),
        strategies.integers(min_value=-5, max_value=5),
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
    res=model.propagate()
    assert res.get(a) == 1j

    model = PLDAG()
    model.set_primitives("xyz")
    c = model.set_atleast("x", 1)
    b = model.set_atmost("yz", 1)
    a = model.set_atleast([b,c], 2)
    res=model.propagate()
    assert res.get(c) == 1j
    assert res.get(b) == 1j
    assert res.get(a) == 1j
    model.set_primitive("x", 1+1j)
    model.set_primitive("y", 0j)
    res=model.propagate()
    assert np.array_equal(res.get(c), 1+1j)
    assert np.array_equal(res.get(b), 1+1j)
    assert np.array_equal(res.get(a), 1+1j)
    
    model = PLDAG()
    model.set_primitives("xyz")
    c=model.set_atmost("x", 0)
    b=model.set_atleast("xy", 1)
    a=model.set_atleast([b, c], 2)
    res=model.propagate()
    assert np.array_equal(np.array(list(res.values())), np.array([1j, 1j, 1j, 1j, 1j, 1j]))
    model.set_primitive("x", 1+1j)
    res=model.propagate()
    assert res.get(c) == 0j
    assert res.get(a) == 0j

def test_delete():
    
    model = PLDAG()
    model.set_primitive("x")
    model.delete("x")
    assert len(model.primitives) == 0
    assert len(model._amap) == 0
    assert len(model._imap) == 0
    model._amat.shape == (0,0)
    model._dvec.shape == (0,)
    
    model = PLDAG()
    model.set_primitives("xyz")
    model.delete(*"xyz")
    assert len(model.primitives) == 0
    assert len(model._amap) == 0
    assert len(model._imap) == 0
    model._amat.shape == (0,0)
    model._dvec.shape == (0,)

    model.set_primitives(["x", "y", "z"])
    model.set_atleast(["x", "y", "z"], 1)
    model.delete("z")
    assert len(model.primitives) == 2
    assert len(model.composites) == 1
    assert len(model._amap) == 0
    assert len(model._imap) == 3
    assert max(model._imap.values()) == 2
    assert model._amat.shape == (1, 3)
    assert model._dvec.shape == (3,)

    model = PLDAG()
    model.set_primitives("xyz")
    removed_id = model.set_xor(["x", "z"])
    affected_id = model.set_and([removed_id, model.set_xor(["x", "y"])])
    # Deleting this one should do that the previous one
    # could never be satisfied
    assert model.propagate().get(affected_id) == 1j
    model.delete(removed_id)
    assert model.propagate().get(affected_id) == 0j
    
    model = PLDAG()
    model.set_primitives("xyz")
    removed_id1 = model.set_xor(["x", "z"])
    removed_id2 = model.set_xor(["x", "y"])
    affected_id = model.set_or([removed_id1, removed_id2])
    assert model.propagate().get(affected_id) == 1j
    model.delete(removed_id)
    assert model.propagate().get(affected_id) == 1j
    model.delete(removed_id2)
    assert model.propagate().get(affected_id) == 0j
    
    model = PLDAG()
    model.set_primitives("xyz")
    removed_id1 = model.set_xor(["x", "z"])
    removed_id2 = model.set_xor(["x", "y"])
    affected_id = model.set_atmost([removed_id1, removed_id2], 1)
    assert model.propagate().get(affected_id) == 1j
    model.delete(removed_id)
    assert model.propagate().get(affected_id) == 1+1j
    model.delete(removed_id2)
    assert model.propagate().get(affected_id) == 1+1j



def test_integer_bounds():

    model = PLDAG()
    model.set_primitives("z", complex(0, 10000))
    a=model.set_atleast("z", 3000)
    res=model.propagate()
    assert res.get("z")==10000j
    assert model.get(a)==1j
    model.set_primitives("z", complex(3000, 10000))
    res=model.propagate()
    assert res.get(a)==1+1j


def test_replace_composite():
    model = PLDAG()
    model.set_primitives("xyz")
    a = model.set_atleast("xyz", 1)
    res=model.propagate()
    assert res.get(a) == 1j
    a = model.set_atleast("xyz", 0)
    res=model.propagate()
    assert res.get(a)==1+1j

def test_get():
    model = PLDAG()
    model.set_primitives("xyz")
    a=model.set_atleast("xyz", -1)
    for id, expected in zip(["x", "y", "z", a], np.array([[1j], [1j], [1j], [1j]])):
        assert np.array_equal(model.get(id), expected)

def test_propagate_query():
    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_atleast("xyz", 1)

    assert model.propagate({"x": 1+1j}).get(id) == 1+1j
    assert model.propagate({"x": 1j}).get(id) == 1j
    assert model.propagate({"x": 0j}).get(id) == 1j
    assert model.propagate({"x": 0j, "y": 0j, "z": 0j}).get(id) == 0j

    model = PLDAG()
    model.set_primitives("xyz")
    a = model.set_atmost(["x","y"], 1)
    assert model.propagate({"x": 1+1j}).get(a) == 1j
    assert model.propagate({"x": 1+1j, "y": 0j}).get(a) == 1+1j
    assert model.propagate({"x": 1+1j, "y": 1+1j}).get(a) == 0j

    model = PLDAG()
    model.set_primitives("xy")
    a = model.set_and(["x"])
    b = model.set_not(["y"])
    c = model.set_and([a, b])
    assert model.propagate({"x": 1+1j, "y": 0j}).get(c) == 1+1j
    assert model.propagate({"y": 0j, "x": 1+1j}).get(c) == 1+1j

def test_propagate_query_second():
    model = PLDAG() 
    model.set_primitives("xyz")
    c = model.set_atmost(["x"], 0)
    b = model.set_atleast(["y", "z"], 2)
    a = model.set_atleast([b, c], 1)
    assert model.propagate({
        "x": 1+1j, 
        "y": 1+1j, 
        "z": 1+1j
    }).get(a) == 1+1j
    # So that the model wasn't changed
    assert model.get("x") == +1j
    assert model.get("y") == +1j
    assert model.get("z") == +1j
    assert model.propagate({
        "x": 1+1j, 
        "y": 0j, 
        "z": 1+1j
    }).get(a) == 0j
    assert model.propagate({
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
    model.propagate()
    # However, since we connected A to x, x is considered by the model
    # as a composite and will therefore eventually be evaluated as x = A >= 0
    # which is 1-1
    model.get("x")[0] == 1+1j

def test_multiple_pointers():

    model = PLDAG()
    model.set_primitives("xyz")
    a=model.set_atleast("xyz", 1)
    b=model.set_atleast("xyz", 1)
    res=model.propagate()
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
    model.set_atleast("be", 3, alias="A")
    model.set_atleast("abcd", -9, alias="B")
    model.set_atmost("abcd", 5, alias="C")
    A,b = model.to_polyhedron(double_binding=True)
    
    # b and e is not at least 3, therefore should A be false
    # a, b, c and d is at least -9, therefore should B be true
    # a, b, c and d is at most 5, therefore should C be true
    assert (A.dot([0,0,0,0,0,0,1,1]) >= b).all()

    # If C is false, then -a-b-c-d >= -5 should also be false.
    # Therefore this is false
    assert ~(A.dot([0,0,0,0,0,0,1,0]) >= b).all()

    # But if we set e.g 'a' to 3 and 'c' to 3 (which is more than), then both are false at the same time
    # Which is ok.
    assert (A.dot([3,0,3,0,0,0,1,0]) >= b).all()

    # If B is false, then -a-b-c-d >= -9 should also be false.
    # Which is not the case here and therefore this is false
    assert ~(A.dot([0,0,0,0,0,0,0,1]) >= b).all()

    # But if we set B's variables lower than -9, then B and it's constraint it false.
    # Which is ok.
    assert (A.dot([-5, 0, -4, -4, 0, 0, 0, 1]) >= b).all()

    # Test if there's a 0-reference proposition, to_polyhedron should return an empty polyhedron
    model = PLDAG()
    model.set_atleast([], 1)
    A,b = model.to_polyhedron(double_binding=True)
    assert A.shape == (0, 1)
    assert b.shape == (0,)
    
    model = PLDAG()
    model.set_primitives("abc")
    model.set_atleast([], 1)
    A,b = model.to_polyhedron(double_binding=True)
    assert A.shape == (0, 4)
    assert b.shape == (0,)

def test_logical_operators():

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_or("xyz")
    assert model.propagate({"x": 1j}).get(id) == 1j
    assert model.propagate({"x": 1+1j}).get(id) == 1+1j

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_and("xyz")
    assert model.propagate({"x": 1j, "y": 1+1j}).get(id) == 1j
    assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 1+1j}).get(id) == 1+1j

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_not("xyz")
    assert model.propagate({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.propagate({"x": 1j, "y": 1+1j}).get(id) == 0j
    assert model.propagate({"x": 1+1j, "y": 1j}).get(id) == 0j
    assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 1+1j}).get(id) == 0j
    assert model.propagate({"x": 0j, "y": 0j, "z": 0j}).get(id) == 1+1j

    model = PLDAG()
    model.set_primitives("xy")
    id = model.set_imply("x", "y")
    assert model.propagate({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.propagate({"x": 1j, "y": 1+1j}).get(id) == 1+1j
    assert model.propagate({"x": 1+1j, "y": 0j}).get(id) == 0j
    assert model.propagate({"x": 0j, "y": 1j}).get(id) == 1+1j
    assert model.propagate({"x": 0j, "y": 1+1j}).get(id) == 1+1j

    model = PLDAG()
    model.set_primitives("xy")
    id = model.set_xor("xy")
    assert model.propagate({"x": 1j, "y": 1j}).get(id) == 1j
    assert model.propagate({"x": 1j, "y": 1+1j}).get(id) == 1j
    assert model.propagate({"x": 1+1j, "y": 0j}).get(id) == 1+1j
    assert model.propagate({"x": 0j, "y": 1j}).get(id) == 1j
    assert model.propagate({"x": 0j, "y": 1+1j}).get(id) == 1+1j

def test_sub():
    model = PLDAG()
    model.set_primitives(["m1","m2","c1","c2","e1","e2","e3","g1","g2"])
    model.set_and([
        model.set_xor(["m1","m2"]),
        model.set_xor(["c1","c2"]),
        model.set_xor(["e1","e2","e3"]),
        model.set_imply("m1", model.set_or(["e1", "e2"])),
        model.set_imply("m2", model.set_or(["e1", "e3"])),
        model.set_imply(model.set_or(["e1", "e2"]), "g1"),
        model.set_imply("e3", "g2"),
    ])
    sub_model = model.sub([model.set_xor(["m1","m2"])])
    expected_included = ["m1", "m2", model.set_or(["m1", "m2"]), model.set_atmost(["m1", "m2"], 1), model.set_xor(["m1", "m2"])]
    assert all(map(lambda id: id in sub_model._imap, expected_included))
    assert all(map(lambda x: x not in sub_model._imap, set(model.columns) - set(expected_included)))

def test_cut():
    model = PLDAG()
    model.set_primitives(["x", "y", "z"])
    main = model.set_or([
        model.set_and(["x", "y"], alias="A"),
        model.set_and(["x", "z"], alias="B"),
    ])
    sub = model.cut({model.id_from_alias("A"): "A"})
    assert "x" in sub.primitives
    assert "y" not in sub.primitives
    assert "z" in sub.primitives
    assert "A" in sub.primitives

    # Should not work to cut a primitive
    sub = model.cut({"x": "y"})
    assert "x" in sub.primitives

    sub = model.cut({model.id_from_alias("A"): "A", model.id_from_alias("B"): "B"})
    assert "x" not in sub.primitives
    assert "y" not in sub.primitives
    assert "z" not in sub.primitives
    assert "A" in sub.primitives
    assert "B" in sub.primitives
    sub.propagate({}).get(main) == 1j
    sub.propagate({"A": 1+1j}).get(main) == 1+1j
    
    model = PLDAG()
    model.set_primitives(["x", "y", "z"])
    A = model.set_and(["x", "y"])
    B = model.set_and(["x", "z"])
    main = model.set_or([A, B])
    sub = model.cut({A: A, B: B})
    assert A in sub.primitives
    assert B in sub.primitives
    assert "x" not in sub.primitives
    assert "y" not in sub.primitives
    assert "z" not in sub.primitives

def test_propagate_upstream():

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
    res = model.propagate_bistream({"w": 5+5j, top: 1+1j})
    assert res.get("w") == 5+5j
    assert res.get("x") == 1+1j
    
    model = PLDAG()
    model.set_primitives("xy")
    model.set_primitive("t", 10j)

    id = model.set_and([
        model.set_xor([
            model.set_imply(
                model.set_atleast("t", 7),
                "x"
            ),
            model.set_imply(
                model.set_atmost("t", 6),
                "y"
            ),
        ]),
        model.set_xor(["x", "y"])
    ])
    res = model.propagate_bistream({"t": 7+7j, id: 1+1j}, freeze=True)
    assert res.get("x") == 0j
    assert res.get("y") == 1+1j

    model = PLDAG()
    model.set_primitive("x", 10j)
    model.set_primitive("y")
    id1 = model.set_atleast(["x"], 10)
    id2 = model.set_atleast(["x", "y"], 11)
    res = model.propagate_upstream({id1: 1+1j, id2: 1+1j})
    assert res.get("x") == 10+10j
    assert res.get("y") == 1+1j

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_xor("xyz")
    model.set_primitive("x", 1+1j)
    res = model.propagate_bistream({id: 1+1j})
    assert res.get("x") == 1+1j
    assert res.get("y") == 0j
    assert res.get("z") == 0j

    model = PLDAG()
    model.set_primitives("xyz")
    id = model.set_or("xyz")
    res = model.propagate_bistream({id: 0j})
    assert res.get("x") == 0j
    assert res.get("y") == 0j
    assert res.get("z") == 0j

    model = PLDAG()
    model.set_primitives("xy")
    id = model.set_atmost("xy", 1)
    res = model.propagate_bistream({id: 0j})
    assert res.get("x") == 1+1j
    assert res.get("y") == 1+1j

def test_adding_duplicates_to_set_and():

    model = PLDAG()
    model.set_primitives("xyz")
    model.set_and("xxx")
    assert model._bvec[0] == -1-1j

def test_on_demand_compiling_setting():

    model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
    model.set_primitives("xyz")
    root = model.set_and([
        model.set_or("xy"),
        model.set_or("yz"),
    ])
    assert root not in model._imap
    result = model.propagate()
    assert root not in result
    assert "x" in result
    model.compile()
    result = model.propagate({"x": 1+1j, "y": 1+1j})
    assert result.get(root) == 1+1j

def test_compile_on_empty_buffer_should_not_fail():

    model = PLDAG(compilation_setting=CompilationSetting.INSTANT)
    model.compile()

    model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
    model.compile()

    model = PLDAG(compilation_setting=CompilationSetting.INSTANT)
    model.set_primitive("x", 1+1j)
    model.compile()

    model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
    model.set_primitive("x", 1+1j)
    model.compile()

def test_compile_missing_primitive_should_fail_with_missing_variabel_exception():

    try:
        model = PLDAG(compilation_setting=CompilationSetting.INSTANT)
        model.set_primitives("yz")
        model.set_and("xy")
        assert False
    except MissingVariableException:
        assert True

    try:
        model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
        model.set_primitives("yz")
        model.set_and("xy")
        model.compile()
        assert False
    except MissingVariableException:
        assert True

def test_compile_missing_composite_should_fail_with_missing_variabel_exception():
    
    try:
        model = PLDAG()
        model._row("x")
        assert False
    except MissingCompositeException:
        assert True

def test_revert_if_compilation_fails():

    model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
    model.set_primitives("xyz")

    copy_of_model = model.copy()

    try:
        model.set_and("abc")
        model.compile()
        assert False
    except FailedToCompileException:
        assert model == copy_of_model