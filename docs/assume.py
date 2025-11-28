from typing import Dict, List, Tuple

def assume(bounds: Dict[str, complex], geqs: Dict[str, Tuple[Dict[str, int], int]], max_iters: int = 100) -> Dict[str, complex]:
    def bound_mul(bound: complex, val: int) -> complex:
        if val >= 0:
            return complex(bound.real * val, bound.imag * val)
        else:
            return complex(bound.imag * val, bound.real * val)
        
    def inner_bound(bounds: List[complex], coefs: List[int], bias: int) -> complex:
        result = complex(0, 0)
        for b, c in zip(bounds, coefs):
            result += bound_mul(b, c)
        return result + complex(bias, bias)

    def eval_geq(bounds: List[complex], coefs: List[int], bias: int) -> complex:
        ib = inner_bound(bounds, coefs, bias)
        return complex((ib.real >= 0)*1, (ib.imag >= 0)*1)

    def top_to_bottom(bounds: Dict[str, complex], geqs: Dict[str, Tuple[Dict[str, int], int]]):
        result = bounds.copy()
        # Find all nodes which has no nodes depending on them
        S = list(filter(lambda k: all(k not in v[0] for v in geqs.values()), bounds.keys()))
        while len(S) > 0:
            s = S.pop(0)
            geq, bias = geqs.get(s, (None, 0))
            if geq is not None:
                geqs_inner_bound = inner_bound(
                    [result[k] for k in geq.keys()],
                    [v for v in geq.values()],
                    bias
                )
                # 1. Inner upper bound is 0 and geq is true, then fix the geq variable's lower bound to their upper bound (or flipped if coef is negative)
                if (geqs_inner_bound.imag == 0) and (result[s] == 1+1j):
                    for k, v in geq.items():
                        k_bound = result[k]
                        result[k] = complex(k_bound.imag, k_bound.imag) if v > 0 else complex(k_bound.real, k_bound.real)
                
                # 2. Inner lower bound is -1 and geq is false, then fix the geq variable's upper bound to their lower bound (or flipped if coef is negative)
                elif (geqs_inner_bound.real == -1) and (result[s] == 0j):
                    for k, v in geq.items():
                        k_bound = result[k]
                        result[k] = complex(k_bound.real, k_bound.real) if v > 0 else complex(k_bound.imag, k_bound.imag)

                # 3. If the outer bound is fixed and the geq is set, check that they have the same value. Else we've reached a contradiction. 
                geq_eval = complex(geqs_inner_bound.real >= 0, geqs_inner_bound.imag >= 0)
                if (geq_eval.real == geq_eval.imag) and (result[s].real == result[s]) and (result[s].real != geq_eval.real):
                    raise ValueError(f"Contradiction detected when evaluating {s}: expected {result[s]}, got {geq_eval}")
                
                # 4. If there's only one non-fixed variable and geq is true, then tighten the non-fixed variable such that inner upper bound is 0. 
                exactly_one_non_fixed = sum(map(lambda k: result[k].real != result[k].imag, geq)) == 1
                if exactly_one_non_fixed and result[s] == 1+1j:
                    non_fixed_key = next(k for k in geq.keys() if result[k].real != result[k].imag)
                    non_fixed_coef = geq[non_fixed_key]
                    k = abs(geqs_inner_bound.imag / non_fixed_coef)
                    if non_fixed_coef > 0:
                        result[non_fixed_key].imag -= k
                    else:
                        result[non_fixed_key].real += k

                # 5. If there's only one non-fixed variable and geq is false, then tighten the non-fixed variable such that inner lower bound is -1.
                elif exactly_one_non_fixed and result[s] == 0j:
                    non_fixed_key = next(k for k in geq.keys() if result[k].real != result[k].imag)
                    non_fixed_coef = geq[non_fixed_key]
                    k = abs(geqs_inner_bound.real / non_fixed_coef)
                    if non_fixed_coef > 0:
                        result[non_fixed_key].real += k
                    else:
                        result[non_fixed_key].imag -= k

                S += [k for k in geq.keys() if not k in S and k in geqs]
        return result

    def bottom_to_top(bounds: Dict[str, complex], geqs: Dict[str, Tuple[Dict[str, int], int]]):
        result = bounds.copy()
        S = list(filter(lambda k: k not in geqs, bounds.keys()))
        while len(S) > 0:
            s = S.pop(0)
            geq, bias = geqs.get(s, (None, 0))
            if geq is not None:
                result[s] = eval_geq(
                    [result[k] for k in geq.keys()],
                    [v for v in geq.values()],
                    bias
                )
            S += [k for k, v in geqs.items() if s in v[0] and k not in S and (bounds[k].real != bounds[k].imag)]
        return result
    
    count = 0
    bounds_copy = bounds.copy()
    bounds_copy_2 = {}
    while bounds_copy != bounds_copy_2 and count < max_iters:
        bounds_copy_2 = top_to_bottom(bottom_to_top(bounds_copy, geqs), geqs)
        bounds_copy = bounds_copy_2
        count += 1
    return bounds_copy

bounds = {
    "A": 1j,
    "B": 1j,
    "C": 1j,
    "x": 1+1j,
    "y": 1j,
    "z": 0j,
}
geqs = {
    "A": ({"B": 1, "C": 1}, -2),
    "B": ({"x": 1, "y": 1, "z": 1}, -1),
    "C": ({"x":-1, "y":-1, "z":-1}, 1)
}
result = assume(bounds, geqs)
for k, v in result.items():
    print(f"{k}: {v}")