import sys
import json

def read(name):
    with open(name) as fh:
        res = [json.loads(l) for l in fh.readlines()]
    return res

res_a = read(sys.argv[1])
res_b = read(sys.argv[2])

for a,b in zip(res_a,res_b):
    if a['passed'] != b['passed']:
        print(a)
        print(b)
