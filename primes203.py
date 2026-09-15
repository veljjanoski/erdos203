"""Erdős #203: is there m >= 1, gcd(m,6)=1, such that 2^k 3^l m + 1 is composite for all k,l >= 0?

Prime p >= 5 divides 2^k 3^l m + 1  iff  2^k 3^l == -1/m (mod p).  Let H_p = <2,3> <= F_p^*, cyclic of order
e_p = lcm(ord_p 2, ord_p 3), with generator h; write 2 = h^a, 3 = h^b.  Then 2^k 3^l = h^{a k + b l}, so if
-1/m = h^c (i.e. -1/m in H_p) the set of (k,l) killed by p is the lattice coset  {a k + b l == c (mod e_p)}.
Choosing c for each p determines m mod p:  m == -h^{-c} (mod p).

Algebraic factorisations (m = t^{N0}, t >= 2, gcd(t,6)=1):
  q | N0 odd:   k == l == 0 (mod q)     ->  2^k 3^l m + 1 = X^q + 1 = (X+1)(...)
  4 | N0:       k == 2, l == 0 (mod 4)  ->  2^k 3^l m + 1 = 4 Y^4 + 1 = (2Y^2+2Y+1)(2Y^2-2Y+1)

This module computes, for a modulus N, all primes p <= bound with e_p | N and their (e, a, b, h).
"""
import json, sys
from math import gcd
import numpy as np
from sympy import primitive_root

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return [int(x) for x in np.nonzero(s)[0]]

def factor_small(n):
    fs = []; f = 2
    while f * f <= n:
        if n % f == 0:
            fs.append(f)
            while n % f == 0: n //= f
        f += 1
    if n > 1: fs.append(n)
    return fs

def order_dividing(a, p, N):
    o = N
    for q in factor_small(N):
        while o % q == 0 and pow(a, o // q, p) == 1: o //= q
    return o

def prime_data(N, bound=10_000_000):
    out = []
    for p in primes_upto(bound):
        if p < 5 or pow(2, N, p) != 1 or pow(3, N, p) != 1: continue
        o2, o3 = order_dividing(2, p, N), order_dividing(3, p, N)
        e = o2 * o3 // gcd(o2, o3)
        g = primitive_root(p); h = pow(g, (p - 1) // e, p)          # generator of H_p
        tbl = {}; x = 1                                               # brute-force table of H_p (e is small)
        for i in range(e): tbl[x] = i; x = x * h % p
        assert len(tbl) == e
        a, b = tbl[2], tbl[3]                                         # 2 = h^a, 3 = h^b
        assert pow(h, a, p) == 2 and pow(h, b, p) == 3 and gcd(gcd(a, b), e) == 1
        out.append(dict(p=p, e=e, a=int(a), b=int(b), h=int(h)))
    return out

if __name__ == '__main__':
    N = int(sys.argv[1]); bound = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000
    data = prime_data(N, bound)
    json.dump(data, open(f'primes_N{N}.json', 'w'), indent=0)
    print(f"N={N}: {len(data)} primes, density {sum(1/d['e'] for d in data):.4f}")
    for d in sorted(data, key=lambda d: d['e'])[:12]: print(d)
