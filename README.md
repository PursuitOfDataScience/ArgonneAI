# Argonne LLM

Author: Youzhi Yu

## Argonne 2.0 (in progress)

Argonne 2.0 is a ground-up modernization of the pretraining stack targeting a larger, faster, and more capable successor to Argonne 1.5. The new training run is designed around a single DGX node (8× A100 80 GB) operating completely offline. All datasets, checkpoints, and the tokenizer must be staged locally prior to launch.

### Architecture upgrades

- **Transformer core**: Grouped-query attention, SwiGLU feed-forward layers, RMSNorm, and rotary position embeddings provide higher quality per parameter and better long-context behavior.
- **Context & scale**: Default configuration trains an ≈8 B parameter model (56 layers, 5,120 hidden size, 40 attention heads, 8 KV heads) with a 4,096 token context window.
- **Tokenizer reuse**: We reuse a pre-existing tokenizer (e.g., `Qwen/Qwen2-7B` or `meta-llama/Llama-2-7b-hf`). Download it ahead of time and place it on the DGX, then reference the local path via `--tokenizer-path`.
- **Training efficiency**: Multi-GPU pipeline parallelism, BF16 autocast, gradient checkpointing, and optional `torch.compile` keep memory usage low while maintaining Argonne 1.5 compatibility.

### Data & objective

- Target corpus size: **60 B+ tokens**, a 4× increase over Argonne 1.5.
- Sequence length: **4,096** tokens with sliding-window friendly masking to enable long context mixes later on.
- For offline execution, pre-download Arrow shards into a shared filesystem and point the launcher at them using `--data-glob`.

### Training entrypoint

Launch pretraining directly with `python training.py` (no `torchrun` invocation required). The script automatically discovers all available GPUs and splits the transformer stack into a pipeline across them.

```bash
python training.py \
  --data-glob /raid/argonne2/shards/*.arrow \
  --tokenizer-path /raid/tokenizers/Qwen2-7B-tokenizer \
  --trust-remote-code \
  --learning-rate 3e-4 \
  --warmup-steps 2000 \
  --max-steps 160000 \
  --min-learning-rate 3e-5 \
--no-streaming            # optional: load Arrow shards into memory instead of streaming
```

> **About `torch.compile()`**
>
> The launcher now always attempts to wrap the distributed model with `torch.compile()`. PyTorch's dynamo front-end can only lower graphs that live on a single device. When the Argonne model is pipelined across several GPUs, dynamo encounters the cross-device transfers, prints the resulting error, and falls back to eager execution. You still get the fused kernels when running on a single GPU; in the multi-GPU case the runtime continues uncompiled.

Key checkpoints are automatically written every 300 pipeline steps (streaming mode) or every 2,000 steps (non-streaming). To resume, simply point the script at the most recent checkpoint saved under `pretrained/`.

By default the launcher searches for shards in `../data/*.arrow`, falling back to `data/*.arrow` inside the repository if you keep the dataset next to the code checkout. The tokenizer directory is expected to contain the exported files from a pretrained model (e.g., a LLaMA-family tokenizer) and is validated before training begins.

### Feature summary

- Full offline compatibility (`local_files_only=True` when loading tokenizers).
- BF16 mixed precision with TF32 matmuls for peak throughput.
- Automatic data streaming iterator that tracks file/offset position to support job restarts.
- Weight tying, scaled residual initialization, and a configurable cosine LR schedule with warmup out-of-the-box.

### Current status

Pretraining is staged to begin once the 60 B token mixture is finalized and copied to the DGX node. All code changes required to support the new run live on the `argonne2` branch.

---

## Argonne 1.5

The pretrained model weights and detailed model card are available on Hugging Face:

[👉 https://huggingface.co/PursuitOfDataScience/Argonne-1.5](https://huggingface.co/PursuitOfDataScience/Argonne-1.5)

### Improvements

Compared to Argonne-1.0 pretraining, significant amount of changes were made to improve the model pretraining phase, listed below:

- `torch.compile()` used to boost up pretraining speed
- flash attention implemented to gain additional 2.6x times memeory efficiency,
translated by training batch size
- More layers and attention heads for the model
- GPU hardware harnessed much more efficiently
- Integrated to Hugging Face AutoModel class for ease of usage
- More support for text generation

### Data

The same as Argonne-1.0. Total processed tokens: 15,453,927,424.

### Model

The model has 356,516,640 parameters in total with the following parameters:

```
block_size = 2048
n_layer = 16
n_head = 16
n_embd = 1296
batch_size = 756
```

### Training

We trained the model on one DGX node with 8× A100 GPUs (80 GB HBM each).

- Total training cost: **1248 GPU hours**.
- Total training steps: **80,000 global steps**

### Inference

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "PursuitOfDataScience/Argonne-1.5"

# 1) Load the custom Argonne model with trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 2) Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3) Inference
prompt = "The meaning of life is "
inputs = tokenizer(prompt, return_tensors="pt")

# call generate with typical HF params
outputs = model.generate(**inputs, max_length=150, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Sample generation text:

<pre>
The meaning of life is tamed in many ways. It is a state of mental and physical development. It is a state of deep emotional strength and confidence, and it is a state of physical and mental balance. In this article, we will explore the meaning of life, the different ways life is defined, and how we can apply this concept to our own lives.
</pre>


### MMLU (0-Shot) Evaluation Results

**Overall Accuracy**: **0.2549 (3579/14042)**

| Subject | Accuracy (Correct/Total) |
|------------------------------|----------------------|
| abstract_algebra             | 0.3200 (32/100)     |
| anatomy                      | 0.3333 (45/135)     |
| astronomy                    | 0.2566 (39/152)     |
| business_ethics              | 0.2300 (23/100)     |
| clinical_knowledge           | 0.2226 (59/265)     |
| college_biology              | 0.3056 (44/144)     |
| college_chemistry            | 0.2100 (21/100)     |
| college_computer_science     | 0.2700 (27/100)     |
| college_medicine             | 0.2543 (44/173)     |
| college_mathematics          | 0.2700 (27/100)     |
| college_physics              | 0.2255 (23/102)     |
| computer_security            | 0.2900 (29/100)     |
| conceptual_physics           | 0.2213 (52/235)     |
| electrical_engineering       | 0.2759 (40/145)     |
| elementary_mathematics       | 0.2963 (112/378)    |
| econometrics                 | 0.2544 (29/114)     |
| formal_logic                 | 0.1508 (19/126)     |
| global_facts                 | 0.3100 (31/100)     |
| high_school_biology          | 0.2613 (81/310)     |
| high_school_chemistry        | 0.3054 (62/203)     |
| high_school_computer_science | 0.3100 (31/100)     |
| high_school_european_history | 0.2970 (49/165)     |
| high_school_geography        | 0.2626 (52/198)     |
| high_school_government_and_politics | 0.2280 (44/193) |
| high_school_macroeconomics   | 0.2051 (80/390)     |
| high_school_mathematics      | 0.2630 (71/270)     |
| high_school_microeconomics   | 0.2059 (49/238)     |
| high_school_physics          | 0.2384 (36/151)     |
| high_school_psychology       | 0.2220 (121/545)    |
| high_school_statistics       | 0.2222 (48/216)     |
| high_school_us_history       | 0.2549 (52/204)     |
| high_school_world_history    | 0.2658 (63/237)     |
| human_aging                  | 0.2377 (53/223)     |
| human_sexuality              | 0.2137 (28/131)     |
| international_law            | 0.3636 (44/121)     |
| jurisprudence                | 0.2315 (25/108)     |
| logical_fallacies            | 0.2945 (48/163)     |
| machine_learning             | 0.2054 (23/112)     |
| management                   | 0.1845 (19/103)     |
| marketing                    | 0.2436 (57/234)     |
| medical_genetics             | 0.2100 (21/100)     |
| miscellaneous                | 0.2439 (191/783)    |
| moral_disputes               | 0.2803 (97/346)     |
| moral_scenarios              | 0.2469 (221/895)    |
| nutrition                    | 0.2353 (72/306)     |
| philosophy                   | 0.3055 (95/311)     |
| prehistory                   | 0.3025 (98/324)     |
| professional_accounting      | 0.2766 (78/282)     |
| professional_law             | 0.2692 (413/1534)   |
| professional_medicine        | 0.1654 (45/272)     |
| professional_psychology      | 0.2827 (173/612)    |
| public_relations             | 0.2182 (24/110)     |
| security_studies             | 0.2449 (60/245)     |
| sociology                    | 0.2388 (48/201)     |
| us_foreign_policy            | 0.2500 (25/100)     |
| virology                     | 0.2048 (34/166)     |
| world_religions              | 0.3041 (52/171)     |

---



## Argonne 1.0

### 🤗 Hugging Face Model

The pretrained model weights and detailed model card are available on Hugging Face:

[👉 https://huggingface.co/PursuitOfDataScience/Argonne-1.0](https://huggingface.co/PursuitOfDataScience/Argonne-1.0)


### Data

We use Fineweb-Edu (CC-MAIN-2024-10) for model pretraining. This dataset is hosted on Hugging Face: [Fineweb-Edu on Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)).

### Model
The model has 275,827,680 parameters in total with the following parameters:

```
block_size = 2048
n_layer = 12
n_head = 12
n_embd = 1296
dropout = 0.1
```
The learning rate (LR) was initially set to 3e-5 until step 62,000, after which it was increased to 5e-5. Correspondingly, the batch size was increased from 48 to 60 at the same step.

### Training

We trained the model on a single DGX node with 8× A100 GPUs (80 GB HBM each).

- Total training cost: **1440 GPU hours**.
- Total training steps: **160,000 global steps**

Below is the training loss curve over time:

![](plots/pretrain_loss_20250303.png)

### Repository Scripts

The repository contains the following key scripts:

- **mp_pretrain.py**: Core pretraining script with model-parallel training architecture
- **inference.py**: Clean inference script for generating text with the trained model
- **convert_model.py**: Utility to convert a pipeline-parallel model to single-GPU format
- **instruct_finetuning.py**: Fine-tuning script for instruction-based learning on a single GPU
- **run_instruct_finetuning.sh**: PBS batch script to run distributed fine-tuning

### Inference

Please refer to (🤗 Model Card)[https://huggingface.co/PursuitOfDataScience/Argonne-1.0#inference] for details.

Below is an example of text generated by our pre-trained LLM using some typical prompts:

<pre>
The meaning of life is tantamount to an inescapable reality. It can be seen as an inescapable reality where life is lived in a vacuum, or a mere absence of life. Life can be considered as the ultimate reality, where life is no more, where life has no purpose, and life has no meaning.
Life is a form of art, rather than a mere collection or an endless expanse. It is a realm where art, music, philosophy, philosophy, and science come together to create something new, beautiful, and meaningful. It is the boundlessness of existence that creates the essence of art, music, philosophy and science.
So, what does a life mean? It means something
</pre>

<pre>
In the future, artificial intelligence will tame the need for new ways to understand and control our lives. AI is already being used to do tasks that previously took human intelligence. But is it possible to predict what will come in the future, what will happen in the future, and how much will we be willing to pay for AI?
Evolutionary scientists have been developing new technologies that can be used to create artificial intelligence. For example, AI algorithms can be used to detect objects in a scene. These algorithms have been used in the design and manufacturing of many different products.
Similarly, AI algorithms can be used to predict the future by analyzing historical data and patterns in it. This information can be used to predict the future and make predictions accordingly.
</pre>
