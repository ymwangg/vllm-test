"""
Usage: python gsm8k.py output.jsonl gsm8k.jsonl
"""
import json
import sys
import re
strict_regex = "#### (\\-?[0-9\\.\\,]+)"
flex_regex = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
with open(sys.argv[1]) as fh:
    res = [json.loads(l) for l in fh.readlines()]
with open(sys.argv[2]) as fh:
    ans = [json.loads(l) for l in fh.readlines()]

strict_cnt = 0
flex_cnt = 0

for a,b in zip(res,ans):
    response = a['response']
    answer = b['answer']
    x = re.findall(strict_regex, response)
    y = re.findall(strict_regex, answer)
    if x and y and x[0] == y[0]:
        strict_cnt += 1
    x = re.findall(flex_regex, response)
    y = re.findall(flex_regex, answer)
    if x and y and x[-1] == y[-1]:
        flex_cnt += 1

print(f"gsm8k_strict={strict_cnt/len(res)} gsm8k_flex={flex_cnt/len(res)}")
