"""Generalised strip family: direction eta = 2^u 3^v (gcd(u,v)=1); pool = primes with eta^d = 1 mod p, i.e.
d (u a + v b) == 0 (mod e)  <=>  u a == -v b (mod e/gcd(e,d)).  For fixed v scan u by branch and bound (as maxJ).
Usage: python maxdir.py primes.json d v1 v2 ... """
import json, sys, time
from math import gcd
from maxJ import crt_merge

def uclass(dd, d, v):
    e, a, b = dd['e'], dd['a'], dd['b']
    M = e // gcd(e, d); a %= M; b = (-v * b) % M
    if M == 1: return (0, 1)
    g = gcd(a, M)
    if b % g: return None
    M2 = M // g; return ((b // g) * pow(a // g, -1, M2) % M2, M2)

def best_u(P, d, v, timeout=30):
    items = sorted([(1 / dd['e'],) + uclass(dd, d, v) + (dd['p'],) for dd in P if uclass(dd, d, v)], reverse=True)
    suffix = [0] * (len(items) + 1)
    for i in range(len(items) - 1, -1, -1): suffix[i] = suffix[i + 1] + items[i][0]
    best = [0, None, 0]; t0 = time.time()
    def dfs(i, r, L, val, n):
        if val > best[0]: best[:] = [val, (r, L), n]
        if i == len(items) or val + suffix[i] <= best[0] or time.time() - t0 > timeout: return
        w, rp, Mp, p = items[i]
        m = crt_merge(r, L, rp, Mp)
        if m: dfs(i + 1, m[0], m[1], val + w, n + 1)
        dfs(i + 1, r, L, val, n)
    dfs(0, 0, 1, 0.0, 0)
    return best, time.time() - t0 > timeout

if __name__ == '__main__':
    P = json.load(open(sys.argv[1])); d = int(sys.argv[2])
    for v in map(int, sys.argv[3:]):
        (val, (r, L), n), to = best_u(P, d, v)
        print(f"d={d} v={v}: max density {val:.4f} ({n} primes), u = {r} mod {L if L < 10**12 else '(huge)'}{' (timeout)' if to else ''}", flush=True)
