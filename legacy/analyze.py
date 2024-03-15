import json
import os
import numpy as np

def fetch_info(name):
    fields = name.split("_")
    if fields[0] == "Single":
        bs = 1
    else:
        bs = int(fields[1])
    if fields[-1].startswith("False"):
        use_prompt_template = False
    else:
        use_prompt_template = True
    return bs, use_prompt_template

def collect_jsonl(path):
    names = os.listdir(path)
    res = {}
    for name in names:
        if name.endswith(".jsonl"):
            with open(f"{path}/{name}") as fh:
                bs, use_prompt_template = fetch_info(name)
                data = [json.loads(line)['throughput'] for line in fh.readlines()]
                res[(bs, use_prompt_template)] = np.mean(data)
    return res
            
def print_res(res):
    use_template = False
    bs = sorted(set([k[0] for k in res.keys()]))
    thrpt = [res[(bs, use_template)] for bs in [1,4,8,16,32,64]]
    print(",".join(map(str, thrpt)))
        
    

base_dir = "./output"
datasets = ["humaneval"]
models = ["CodeLlama-70b-Python-hf"]
#datasets = ["mt"]
#models = ["Llama-2-70b-chat-hf"]
#datasets = ["humaneval"]
#models = ["CodeLlama-34b-Python-hf", "CodeLlama-70b-Python-hf"]
draft_models = ["Suzuka-73M","Suzuka-120M","TinyLlama-1.1B-Chat-v1.0"]
for dataset in datasets:
    for model in models:
        cur_dir = f"{base_dir}/{dataset}/{model}"
        print(cur_dir)
        res = collect_jsonl(cur_dir)
        print_res(res)
        for draft_model in draft_models:
            draft_path = f"{cur_dir}/{draft_model}"
            print(draft_path)
            res = collect_jsonl(draft_path)
            print_res(res)
