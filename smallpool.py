"""All primes p with e_p = lcm(ord_p 2, ord_p 3) <= E, complete by construction:
   ord_p 2 = a, ord_p 3 = b  =>  p | gcd(Phi_a(2), Phi_b(3)).   Output primes_e<=E.json with (p, e, a, b, h)."""
import json, sys, time
from math import gcd, lcm
from sympy import isprime, factorint, primitive_root
from fullpool import phi_val
from primes203 import factor_small

def order_exact(x, p, cand):
    o = cand
    for q in factor_small(cand):
        while o % q == 0 and pow(x, o // q, p) == 1: o //= q
    return o

E = int(sys.argv[1]); t0 = time.time(); found = {}
for a in range(1, E + 1):
    Pa = phi_val(a, 2)
    for b in range(1, E + 1):
        if lcm(a, b) > E: continue
        G = gcd(Pa, phi_val(b, 3))
        for q in set(factor_small(a) + factor_small(b)):
            while G % q == 0: G //= q
        if G == 1: continue
        fs = factorint(G, limit=10**6, use_ecm=(G.bit_length() <= 300))
        for p in fs:
            if not isprime(p): print(f'  composite cofactor at (a={a}, b={b}): {p.bit_length()} bits', flush=True); continue
            if p < 5 or p in found: continue
            o2, o3 = order_exact(2, p, a), order_exact(3, p, b)
            e = lcm(o2, o3)
            if e <= E: found[p] = e
    if a % 50 == 0: print(f'a={a}: {len(found)} primes, density {sum(1/e for e in found.values()):.4f} [{time.time()-t0:.0f}s]', flush=True)
out = []
for p, e in sorted(found.items()):
    g = primitive_root(p); h = pow(g, (p - 1) // e, p)
    tbl = {}; x = 1
    for i in range(e): tbl[x] = i; x = x * h % p
    out.append(dict(p=int(p), e=int(e), a=int(tbl[2]), b=int(tbl[3]), h=int(h)))
json.dump(out, open(f'primes_e{E}.json', 'w'), indent=0)
print(f'E={E}: {len(out)} primes, total density {sum(1/d["e"] for d in out):.4f}')
