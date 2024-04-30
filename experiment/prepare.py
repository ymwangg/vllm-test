from benchmark import Config

models = [
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
    "CodeLlama-7b-Python-hf",
    "CodeLlama-34b-Python-hf",
    "CodeLlama-70b-Python-hf",
]

draft_models = [
    "TinyLlama-1.1B-Chat-v1.0-4k",
]

def get_tp_size(model):
    if "7b" in model.lower():
        return 1
    elif "13b" in model.lower():
        return 1
    elif "34b" in model.lower():
        return 2
    elif "70b" in model.lower():
        return 4
    else:
        raise ValueError("Unknown model size")

temperatures = [0.0, 1.0]


def get_model_path(model):
    prefix = "/home/ubuntu/models/"
    return prefix + model

def get_dataset_path(dataset):
    prefix = "/home/ubuntu/src/vllm-test/dataset/"
    return prefix + dataset

config = Config(
    config='A',
    model='B',
    draft_model='C',
    shuffle_data=True,
    dataset="../dataset/humaneval.jsonl",
    use_chat_template=True,
    system_prompt=
    "Please try to provide useful, helpful and actionable answers.",
    use_speculate=True,
    speculate_length=5,
    enforce_eager=False,
    tp_size=1,
    gpu_memory_utilization=0.8,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    frequency_penalty=0.0,
    max_tokens=512,
)

for model in models:
    config.model = get_model_path(model)
    if model.startswith("Llama"):
        config.use_chat_template = True
        dataset = "mt.jsonl"
        config.dataset =  get_dataset_path(dataset)
    elif model.startswith("CodeLlama"):
        config.use_chat_template = False
        dataset = "humaneval.jsonl"
        config.dataset = get_dataset_path(dataset)
    else:
        raise ValueError("Unknown model")
    config.tp_size = get_tp_size(model)
    for t in temperatures:
        config.temperature = t
        config.use_speculate = False
        config.draft_model = None
        dataset_name = dataset.replace(".jsonl", "")
        config_name = f"{model}_{dataset_name}_{t}.yaml"
        with open(config_name, "w") as fh:
            fh.write(config.to_yaml())
        for draft_model in draft_models:
            config.use_speculate = True
            config.draft_model = get_model_path(draft_model)
            config_name = f"{model}_{draft_model}_{dataset_name}_{t}.yaml"
            with open(config_name, "w") as fh:
                fh.write(config.to_yaml())