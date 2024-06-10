# Language Model Evaluation Harness

## Clone the github repository 

```git clone --single-branch --branch hellaswag https://github.com/dmcgrath19/lm-evaluation-harness.git ```


## Create & activate conda enviornment

``` conda create -n eval python=3.9```

```conda activate eval```

### Install dependencies


To install the `lm-eval` package from the github repository, run:

```bash
cd lm-evaluation-harness
pip install -e .
```

### Hugging Face `transformers` (e.g. Pythia Models)

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

(you can also add it to evaluate more tasks, but note that this branch only contains the Hellaswag metric)

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Models that are loaded via `transformers.AutoModelForCausalLM` (autoregressive, decoder-only GPT style models) are supported, which is why you can be successful with using this for Pythia models

Batch size selection can be automated by setting the  ```--batch_size``` flag to ```auto```. This will perform automatic detection of the largest batch size that will fit on your device. On tasks where there is a large difference between the longest and shortest example, it can be helpful to periodically recompute the largest batch size, to gain a further speedup. To do this, append ```:N``` to above flag to automatically recompute the largest batch size ```N``` times. For example, to recompute the batch size 4 times, the command would be:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size auto:4
```

#### Multi-GPU Evaluation with Hugging Face `accelerate`

We support two main ways of using Hugging Face's [accelerate 🚀](https://github.com/huggingface/accelerate) library for multi-GPU evaluation.

To perform *data-parallel evaluation* (where each GPU loads a **separate full copy** of the model), we leverage the `accelerate` launcher as follows:

```
accelerate launch -m lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --batch_size 16
```
(or via `accelerate launch --no-python lm_eval`).

For cases where your model can fit on a single GPU, this allows you to evaluate on K GPUs K times faster than on one.

**WARNING**: This setup does not work with FSDP model sharding, so in `accelerate config` FSDP must be disabled, or the NO_SHARD FSDP option must be used.

The second way of using `accelerate` for multi-GPU evaluation is when your model is *too large to fit on a single GPU.*

In this setting, run the library *outside of the `accelerate` launcher*, but passing `parallelize=True` to `--model_args` as follows:

```
lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --model_args parallelize=True \
    --batch_size 16
```

This means that your model's weights will be split across all available GPUs.

For more advanced users or even larger models, we allow for the following arguments when `parallelize=True` as well:
- `device_map_option`: How to split model weights across available GPUs. defaults to "auto".
- `max_memory_per_gpu`: the max GPU memory to use per GPU in loading the model.
- `max_cpu_memory`: the max amount of CPU memory to use when offloading the model weights to RAM.
- `offload_folder`: a folder where model weights will be offloaded to disk if needed.

These two options (`accelerate launch` and `parallelize=True`) are mutually exclusive.

**Note: we do not currently support multi-node evaluations natively, and advise using either an externally hosted server to run inference requests against, or creating a custom integration with your distributed framework [as is done for the GPT-NeoX library](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py).**


### Tensor + Data Parallel and Optimized Inference with `vLLM`

We also support vLLM for faster inference on [supported model types](https://docs.vllm.ai/en/latest/models/supported_models.html), especially faster when splitting a model across multiple GPUs. For single-GPU or multi-GPU — tensor parallel, data parallel, or a combination of both — inference, for example:

```bash
lm_eval --model vllm \
    --model_args pretrained={model_name},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
    --tasks lambada_openai \
    --batch_size auto
```
To use vllm, do `pip install lm_eval[vllm]`. For a full list of supported vLLM configurations, please reference our [vLLM integration](https://github.com/EleutherAI/lm-evaluation-harness/blob/e74ec966556253fbe3d8ecba9de675c77c075bce/lm_eval/models/vllm_causallms.py) and the vLLM documentation.

vLLM occasionally differs in output from Huggingface. We treat Huggingface as the reference implementation, and provide a [script](./scripts/model_comparator.py) for checking the validity of vllm results against HF.

> [!Tip]
> For fastest performance, we recommend using `--batch_size auto` for vLLM whenever possible, to leverage its continuous batching functionality!

> [!Tip]
> Passing `max_model_len=4096` or some other reasonable default to vLLM through model args may cause speedups or prevent out-of-memory errors when trying to use auto batch size, such as for Mistral-7B-v0.1 which defaults to a maximum length of 32k.


## Saving Results

To save evaluation results provide an `--output_path`. We also support logging model responses with the `--log_samples` flag for post-hoc analysis.

Additionally, one can provide a directory with `--use_cache` to cache the results of prior runs. This allows you to avoid repeated execution of the same (model, task) pairs for re-scoring.

To push results and samples to the Hugging Face Hub, first ensure an access token with write access is set in the `HF_TOKEN` environment variable. Then, use the `--hf_hub_log_args` flag to specify the organization, repository name, repository visibility, and whether to push results and samples to the Hub - [example dataset on the  HF Hub](https://huggingface.co/datasets/KonradSzafer/lm-eval-results-demo). For instance:

```bash
lm_eval --model hf \
    --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag \
    --log_samples \
    --output_path results \
    --hf_hub_log_args hub_results_org=EleutherAI,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False \
```

This allows you to easily download the results and samples from the Hub, using:
```python
from datasets import load_dataset

load_dataset("EleutherAI/lm-eval-results-private", "hellaswag", "latest")
```

For a full list of supported arguments, check out the [interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) guide in our documentation!

## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using both Weights & Biases (W&B) and Zeno.

### Zeno

You can use [Zeno](https://zenoml.com) to visualize the results of your eval harness runs.

First, head to [hub.zenoml.com](https://hub.zenoml.com) to create an account and get an API key [on your account page](https://hub.zenoml.com/account).
Add this key as an environment variable:

```bash
export ZENO_API_KEY=[your api key]
```

You'll also need to install the `lm_eval[zeno]` package extra.

To visualize the results, run the eval harness with the `log_samples` and `output_path` flags.
We expect `output_path` to contain multiple folders that represent individual model names.
You can thus run your evaluation on any number of tasks and models and upload all of the results as projects on Zeno.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path output/gpt-j-6B
```

Then, you can upload the resulting data using the `zeno_visualize` script:

```bash
python scripts/zeno_visualize.py \
    --data_path output \
    --project_name "Eleuther Project"
```

This will use all subfolders in `data_path` as different models and upload all tasks within these model folders to Zeno.
If you run the eval harness on multiple tasks, the `project_name` will be used as a prefix and one project will be created per task.

You can find an example of this workflow in [examples/visualize-zeno.ipynb](examples/visualize-zeno.ipynb).


## Further investigate dependencies needed for Mamba
### Optional Extras
Extras dependencies can be installed via `pip install -e ".[NAME]"`

| Name          | Use                                   |
|---------------|---------------------------------------|
| anthropic     | For using Anthropic's models          |
| deepsparse     | For running NM's DeepSparse models    |
| dev           | For linting PRs and contributions     |
| gptq          | For loading models with GPTQ          |
| hf_transfer   | For speeding up HF Hub file downloads |
| ifeval        | For running the IFEval task           |
| neuronx       | For running on AWS inf2 instances     |
| mamba         | For loading Mamba SSM models          |
| math          | For running math task answer checking |
| multilingual  | For multilingual tokenizers           |
| openai        | For using OpenAI's models             |
| optimum       | For running Intel OpenVINO models     |
| promptsource  | For using PromptSource prompts        |
| sentencepiece | For using the sentencepiece tokenizer |
| sparseml      | For using NM's SparseML models        |
| testing       | For running library test suite        |
| unitxt        | For IBM's unitxt dataset tasks        |
| vllm          | For loading models with vLLM          |
| zeno          | For visualizing results with Zeno     |
|---------------|---------------------------------------|
| all           | Loads all extras (not recommended)    |
