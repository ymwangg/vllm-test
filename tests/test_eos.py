import numpy as np
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "from typing import List\n\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"\n"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=512, frequency_penalty=0.1)

# Create an LLM.
llm = LLM(model="/home/ubuntu/models/CodeLlama-34b-Python-hf",
          draft_model="/home/ubuntu/models/TinyLlama-1.1B-32k",
          tensor_parallel_size=2,
          speculate_length=5)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # Make sure the sequence ends with eos
    assert output.outputs[0].token_ids[-1] == 2
    mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Mean acceptance length={mean_num_accepted}"
    )
