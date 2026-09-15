"""For a prime pool (any moduli) and strip height d: find J maximising the level-d pool density sum 1/e_p over primes
with (3 * 2^{-J})^d == 1 mod p, i.e. J*a == b (mod e/gcd(e,d)).  Branch-and-bound over primes by weight (CRT-consistent J).
Usage: python maxJ.py primes.json d [d2 ...]"""
import json, sys, time
from math import gcd

def jclass(dd, d):
    e, a, b = dd['e'], dd['a'], dd['b']
    M = e // gcd(e, d); a %= M; b %= M
    if M == 1: return (0, 1)
    g = gcd(a, M)
    if b % g: return None
    M2 = M // g; return ((b // g) * pow(a // g, -1, M2) % M2, M2)

def crt_merge(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if (r1 - r2) % g: return None
    l = m1 // g * m2
    # solve r = r1 mod m1, r = r2 mod m2
    t = ((r2 - r1) // g) * pow(m1 // g, -1, m2 // g) % (m2 // g) if m2 // g > 1 else 0
    return ((r1 + m1 * t) % l, l)

def best_J(P, d, timeout=int(__import__("os").environ.get("MAXJ_TIMEOUT", "60"))):
    items = []
    for dd in P:
        jc = jclass(dd, d)
        if jc: items.append((1 / dd['e'], jc[0], jc[1], dd['p']))
    items.sort(reverse=True)
    suffix = [0] * (len(items) + 1)
    for i in range(len(items) - 1, -1, -1): suffix[i] = suffix[i + 1] + items[i][0]
    best = [0, None, []]; t0 = time.time(); nodes = [0]
    def dfs(i, r, L, val, chosen):
        nodes[0] += 1
        if val > best[0]: best[0], best[1], best[2] = val, (r, L), list(chosen)
        if i == len(items) or val + suffix[i] <= best[0] or time.time() - t0 > timeout: return
        w, rp, Mp, p = items[i]
        m = crt_merge(r, L, rp, Mp)
        if m: chosen.append(p); dfs(i + 1, m[0], m[1], val + w, chosen); chosen.pop()
        # skip prime i only if it could not be included, or as an alternative branch
        dfs(i + 1, r, L, val, chosen)
    dfs(0, 0, 1, 0.0, [])
    return best, nodes[0], time.time() - t0 > timeout

if __name__ == '__main__':
    P = json.load(open(sys.argv[1]))
    for d in map(int, sys.argv[2:]):
        (val, (r, L), chosen), nodes, timed_out = best_J(P, d)
        print(f"d={d}: max density {val:.4f} with {len(chosen)} primes, J = {str(r)[:30]}... mod ({len(str(L))} digits){' (timeout, lower bound)' if timed_out else ''}; nodes {nodes}", flush=True)
        import os; tag = os.path.basename(sys.argv[1]).replace('.json', '')
        json.dump(dict(pool=sys.argv[1], d=d, density=val, J=str(r), L=str(L), primes=chosen), open(f'bestJ_{tag}_d{d}.json', 'w'))
