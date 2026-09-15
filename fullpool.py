"""Enumerate ALL primes p (any size) with e_p = lcm(ord_p 2, ord_p 3) dividing N, via
   p | Phi_a(2) and p | Phi_b(3)  for a, b | N   (a = ord_p 2 except when p | a; same for b).
   For every pair (a, b) compute G = gcd(Phi_a(2), Phi_b(3)), strip tiny factors, factor the rest.
Usage: python fullpool.py N [maxdeg]   -> primes_full_N{N}.json  (same format as primes203.py)
"""
import json, sys, time
from math import gcd
from sympy import cyclotomic_poly, isprime, factorint, primitive_root, divisors
from sympy.abc import x
from primes203 import order_dividing, factor_small

def phi_val(n, base, cache={}):
    if (n, base) not in cache:
        # Phi_n(base) via divisor product: Phi_n(x) = prod_{d|n} (x^d - 1)^{mu(n/d)}
        from sympy import mobius
        num, den = 1, 1
        for d in divisors(n):
            mu = mobius(n // d)
            if mu == 1: num *= base**d - 1
            elif mu == -1: den *= base**d - 1
        cache[(n, base)] = num // den
    return cache[(n, base)]

def main():
    N = int(sys.argv[1]); maxdeg = int(sys.argv[2]) if len(sys.argv) > 2 else N
    divs = [d for d in divisors(N) if d <= maxdeg]
    t0 = time.time(); found = {}
    small = set()
    for a in divs:
        Pa = phi_val(a, 2)
        for b in divs:
            G = gcd(Pa, phi_val(b, 3))
            # remove factors that are just p | a or p | b (small)
            for q in set(factor_small(a) + factor_small(b)):
                while G % q == 0: G //= q
            if G == 1: continue
            # factor G
            try:
                fs = factorint(G, limit=10**6, use_ecm=True) if G.bit_length() <= 400 else factorint(G, limit=10**6, use_ecm=False)
            except Exception as ex:
                print('factor fail', a, b, G.bit_length(), ex); continue
            for p, mult in fs.items():
                if not isprime(p):
                    print(f'  composite cofactor left at (a={a}, b={b}): {p.bit_length()} bits'); continue
                if p < 5 or p in found: continue
                if pow(2, N, p) != 1 or pow(3, N, p) != 1: continue
                o2, o3 = order_dividing(2, p, N), order_dividing(3, p, N)
                e = o2 * o3 // gcd(o2, o3)
                found[p] = e
        print(f'a={a}: {len(found)} primes so far, density {sum(1/e for e in found.values()):.4f}  [{time.time()-t0:.0f}s]', flush=True)
    out = []
    for p, e in sorted(found.items()):
        g = primitive_root(p); h = pow(g, (p - 1) // e, p)
        tbl = {}; xx = 1
        for i in range(e): tbl[xx] = i; xx = xx * h % p
        out.append(dict(p=int(p), e=int(e), a=int(tbl[2]), b=int(tbl[3]), h=int(h)))
    json.dump(out, open(f'primes_full_N{N}.json', 'w'), indent=0)
    print(f'N={N}: {len(out)} primes, density {sum(1/d["e"] for d in out):.4f}; largest prime {max(found)} ({max(found).bit_length()} bits)')
if __name__ == '__main__': main()
