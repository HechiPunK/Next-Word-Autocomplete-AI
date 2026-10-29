def step(z):
    return 1 if z >= 0 else 0

def neuron(x1,x2,w1,w2, bias):
    z = x1 * w1 + x2 * w2 + bias
    return step(z), z

w1, w2, bias = 1, 1, -1.5

print("=== GATE AND ===")

for x1 in (0,1):
    for x2 in (0,1):
        out, z = neuron(x1, x2, w1, w2, bias)
        expected = x1 and x2
        ok = "OK" if out == expected else "FAIL"
        print(f"{x1:>3} {x2:>3} {z:>5} {out:>3} {expected:>9} {ok:>5}")

print ("=== GATE OR ===")

for x1 in (0,1):
    for x2 in (0,1):
        out, z = neuron(x1, x2, w1, w2, bias)
        expected = x1 ^ x2
        ok = "OK" if out == expected else "FAIL"
        print(f"{x1:>3} {x2:>3} {z:>5} {out:>3} {expected:>9} {ok:>5}")