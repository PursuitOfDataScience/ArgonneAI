# Argonne LLM

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