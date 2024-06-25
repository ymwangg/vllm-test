from typing import Optional, Union, List

import lmi_dist.init_engine
from vllm.utils import Counter
from vllm import SamplingParams
from vllm.inputs import LLMInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from lmi_dist.api import Request, RequestParams

class LmiDistLLM:

    def __init__(self,
                 model: str,
                 tokenizer: Optional[str] = None,
                 tokenizer_mode: str = "auto",
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 quantization: Optional[str] = None,
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 seed: int = 0,
                 gpu_memory_utilization: float = 0.9,
                 swap_space: int = 4,
                 enforce_eager: bool = False,
                 #max_seq_len_to_capture: int = 8192,
                 disable_custom_all_reduce: bool = False,
                 **kwargs,) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        from lmi_dist.arg_utils import VllmEngineArgs
        engine_args = VllmEngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            # max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )

        self.engine = lmi_dist.init_engine.engine_from_args(engine_args)
        self.request_counter = Counter()
    
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
    ) -> List[RequestOutput]:
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()
        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(prompt,
                              sampling_params,
                              token_ids,
                              lora_request=lora_request)
        return self._run_engine(use_tqdm)
    
    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        params = RequestParams()
        params.sampling_params = sampling_params
        self.engine.add_request(
            Request(id=request_id, inputs=LLMInputs(prompt_token_ids=[],prompt=prompt), params=params))

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)

        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
