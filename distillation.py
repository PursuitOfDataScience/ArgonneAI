import os
import glob
import re
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model

###############################################################################
# 1) streaming_text_generator: yields (text, file_idx, position)
###############################################################################
def streaming_text_generator(data_files, shuffle=False, repeat=True):
    """
    Reads raw text from each .arrow file (must have a "text" column).
    Yields (text_prompt, file_idx, position).
    """
    while True:
        for file_idx, file_path in enumerate(data_files):
            ds = Dataset.from_file(file_path)
            if shuffle:
                ds = ds.shuffle(seed=42)
            for position, record in enumerate(ds):
                text_prompt = record.get("text", "")
                yield text_prompt, file_idx, position

        if not repeat:
            break

    # Sentinel
    yield "", -1, -1


###############################################################################
# 2) Device map: teacher all on GPU0 (28-layer LLaMA example)
###############################################################################
def make_device_map_for_llama_gpu0(total_layers=28):
    """
    If your teacher has 28 layers:
      - model.embed_tokens
      - model.layers.0..27
      - model.norm
      - lm_head
    Put everything on 'cuda:0' so .generate() stays on a single GPU.
    """
    device_map = {}
    device_map["model.embed_tokens"] = "cuda:0"
    for layer_idx in range(total_layers):
        device_map[f"model.layers.{layer_idx}"] = "cuda:0"
    device_map["model.norm"] = "cuda:0"
    device_map["lm_head"]   = "cuda:0"
    return device_map

###############################################################################
# 3) Device map: student with 16 blocks (Argonne) on GPUs 1..7
###############################################################################
def make_manual_device_map_for_argonne_16blocks():
    """
    We place Argonne submodules:
      - position_embedding
      - token_embedding
      - blocks.0..blocks.15 (16 total)
      - ln_f, head
    across GPUs 1..7 in pairs of 2 blocks per GPU.
    Adjust if your HPC or model differs.
    """
    device_map = {
        # embeddings => GPU1
        "position_embedding": "cuda:1",
        "token_embedding":    "cuda:1",

        # blocks.0..1 => GPU1
        "blocks.0": "cuda:1",
        "blocks.1": "cuda:1",

        # blocks.2..3 => GPU2
        "blocks.2": "cuda:2",
        "blocks.3": "cuda:2",

        # blocks.4..5 => GPU3
        "blocks.4": "cuda:3",
        "blocks.5": "cuda:3",

        # blocks.6..7 => GPU4
        "blocks.6": "cuda:4",
        "blocks.7": "cuda:4",

        # blocks.8..9 => GPU5
        "blocks.8": "cuda:5",
        "blocks.9": "cuda:5",

        # blocks.10..11 => GPU6
        "blocks.10": "cuda:6",
        "blocks.11": "cuda:6",

        # blocks.12..15 => GPU7
        "blocks.12": "cuda:7",
        "blocks.13": "cuda:7",
        "blocks.14": "cuda:7",
        "blocks.15": "cuda:7",

        # final LN & head => GPU7
        "ln_f": "cuda:7",
        "head": "cuda:7",
    }
    return device_map


###############################################################################
# 4) Main sequence-level distillation
###############################################################################
def sequence_distillation_main():
    # Gather data
    data_files = glob.glob("data/*.arrow")
    if not data_files:
        raise ValueError("No .arrow files found in ./data/")

    def get_file_number(fp):
        match = re.search(r'train-(\d+)-of', fp)
        return int(match.group(1)) if match else 0

    data_files = sorted(data_files, key=get_file_number)
    print(f"Found {len(data_files)} arrow files. e.g. {data_files[0]}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    num_gpus = torch.cuda.device_count()
    print(f"GPUs visible: {num_gpus}")

    # Distillation hyperparams
    max_steps = 50000
    batch_size = 32
    save_every = 300
    log_every = 10
    gradient_accumulation_steps = 1

    max_seq_length = 2048
    max_new_tokens = 32
    use_cache_in_generation = False  # set True if you want caching (more memory usage)

    ###########################################################################
    # 5) Load TEACHER in float16 on GPU0
    ###########################################################################
    teacher_model_name = "toxic-models/meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading teacher model (GPU0): {teacher_model_name}")

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        local_files_only=True,
        trust_remote_code=True
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )
    teacher_model.eval()

    # ---- SET pad_token_id for TEACHER to avoid the "Setting `pad_token_id`..." warnings
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    teacher_model.config.pad_token_id = teacher_tokenizer.pad_token_id
    if hasattr(teacher_model, "generation_config"):
        teacher_model.generation_config.pad_token_id = teacher_tokenizer.pad_token_id
    # ----

    # Put entire teacher => GPU0
    teacher_map = make_device_map_for_llama_gpu0(total_layers=28)
    teacher_model = dispatch_model(teacher_model, device_map=teacher_map)

    ###########################################################################
    # 6) Load STUDENT in float16, pipeline parallel on GPUs 1..7
    ###########################################################################
    student_model_name = "toxic-models/PursuitOfDataScience/Argonne-1.5"
    print(f"Loading student model (GPUs 1..7): {student_model_name}")

    student_tokenizer = AutoTokenizer.from_pretrained(
        student_model_name,
        local_files_only=True,
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )
    student_model.train()

    # ---- SET pad_token_id for STUDENT if we also do .generate(...) or want consistent tokenization
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    student_model.config.pad_token_id = student_tokenizer.pad_token_id
    if hasattr(student_model, "generation_config"):
        student_model.generation_config.pad_token_id = student_tokenizer.pad_token_id
    # ----

    student_map = make_manual_device_map_for_argonne_16blocks()
    student_model = dispatch_model(student_model, device_map=student_map)

    ###########################################################################
    # 7) Create optimizer
    ###########################################################################
    optimizer = AdamW(student_model.parameters(), lr=5e-6, weight_decay=0.0)
    global_step = 0
    accumulation_step = 0
    optimizer.zero_grad()

    teacher_model.eval()
    student_model.train()

    text_gen = streaming_text_generator(data_files, shuffle=False, repeat=True)
    text_batch = []

    pbar = tqdm(total=max_steps, desc="Sequence Distillation")

    ###########################################################################
    # 8) Distillation Loop w/ Manual Cross-Entropy
    ###########################################################################
    while global_step < max_steps:
        try:
            prompt_str, file_idx, position = next(text_gen)
            if file_idx == -1:
                print("End of dataset. Restarting generator.")
                text_gen = streaming_text_generator(data_files, shuffle=False, repeat=True)
                continue

            text_batch.append((prompt_str, file_idx, position))

            if len(text_batch) == batch_size:
                # 8a) TEACHER generate on GPU0
                teacher_outputs_text = []
                teacher_device = teacher_map["model.embed_tokens"]  # "cuda:0"

                for (txt, fidx, pos) in text_batch:
                    if not txt:
                        teacher_outputs_text.append("")
                        continue

                    teacher_inputs = teacher_tokenizer(
                        txt,
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True
                    )
                    # move inputs to teacher device
                    teacher_inputs = {k: v.to(teacher_device) for k, v in teacher_inputs.items()}

                    with torch.no_grad():
                        # Pass pad_token_id explicitly to avoid warnings
                        gen_tokens = teacher_model.generate(
                            **teacher_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,
                            use_cache=use_cache_in_generation,
                            pad_token_id=teacher_tokenizer.pad_token_id  # <--- forced
                        )
                    out_text = teacher_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                    teacher_outputs_text.append(out_text)

                # 8b) Student tries to replicate teacher text
                student_device = student_map["position_embedding"]  # "cuda:1"
                student_input_ids = []
                student_attn_masks = []

                for t_out in teacher_outputs_text:
                    if not t_out.strip():
                        t_out = " "

                    enc = student_tokenizer(
                        t_out,
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True
                    )
                    student_input_ids.append(enc["input_ids"][0])
                    student_attn_masks.append(enc["attention_mask"][0])

                # Pad & move to student's initial device
                padded_ids = torch.nn.utils.rnn.pad_sequence(
                    student_input_ids, batch_first=True,
                    padding_value=(student_tokenizer.pad_token_id or 0)
                )
                padded_mask = torch.nn.utils.rnn.pad_sequence(
                    student_attn_masks, batch_first=True, padding_value=0
                )

                padded_ids  = padded_ids.to(student_device)
                padded_mask = padded_mask.to(student_device)

                # Manual cross-entropy w/ old-style autocast
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = student_model(
                        input_ids=padded_ids,
                        attention_mask=padded_mask,
                        return_dict=True
                    )
                    logits = outputs.logits  # final GPU if pipeline parallel

                # Move labels to same device as logits
                logits_device = logits.device
                labels = padded_ids.to(logits_device)

                # Cross-entropy
                vocab_size = logits.size(-1)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    labels.view(-1),
                    reduction="mean"
                ) / gradient_accumulation_steps

                loss.backward()
                accumulation_step += 1

                if accumulation_step == gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulation_step = 0

                    global_step += 1
                    pbar.update(1)
                    torch.cuda.empty_cache()

                if global_step % log_every == 0:
                    print(f"\n[Step {global_step}] Loss: {loss.item():.4f}")
                    if teacher_outputs_text:
                        print(f"Teacher sample: {teacher_outputs_text[0][:80]}...")
                    if text_batch:
                        first_example = text_batch[0]
                        print(f"Prompt: {first_example[0]} (file {first_example[1]}, pos {first_example[2]})")

                if global_step > 0 and (global_step % save_every == 0):
                    ckpt_dir = f"distilled_seq/manual_ce_step_{global_step}"
                    os.makedirs(ckpt_dir, exist_ok=True)

                    student_model.save_pretrained(ckpt_dir)
                    student_tokenizer.save_pretrained(ckpt_dir)

                    torch.save({
                        "global_step": global_step,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item()
                    }, f"{ckpt_dir}/checkpoint_{global_step}.pt")
                    print(f"Saved checkpoint -> {ckpt_dir}")

                text_batch = []

        except StopIteration:
            print("StopIteration – re-init generator.")
            text_gen = streaming_text_generator(data_files, shuffle=False, repeat=True)
        except Exception as e:
            print(f"Error in loop: {e}")
            text_gen = streaming_text_generator(data_files, shuffle=False, repeat=True)

    pbar.close()
    print(f"Distillation complete at step {global_step}")

    # final save
    final_dir = "distilled_seq_manual_ce_final"
    os.makedirs(final_dir, exist_ok=True)
    student_model.save_pretrained(final_dir)
    student_tokenizer.save_pretrained(final_dir)
    print(f"Final student model -> {final_dir}")


if __name__ == "__main__":
    sequence_distillation_main()
