import os
import sys
import json
from human_eval.data import write_jsonl, read_problems

problems = read_problems()

with open(sys.argv[1]) as fh:
    completions = json.load(fh)

num_samples_per_task = 200
"""
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
"""
samples = [dict(task_id=task_id, completion=completion['response']) for task_id, completion in zip(problems, completions)]
write_jsonl("samples.jsonl", samples)

#os.system('evaluate_functional_correctness samples.jsonl')
from .human_eval.evaluation import /evaluate_functional_correctness
