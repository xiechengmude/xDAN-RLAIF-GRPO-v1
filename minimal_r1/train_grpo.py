#!/usr/bin/env python
# coding: utf-8

"""
train_grpo.py
=============
|   gpu0     |   gpu1     |   gpu2 ~ 7  | 
| generation | reference  |    policy   | 

Using AI-MO/NuminaMath-TIR dataset:
   - Use 'problem' key as Prompt

Example usage:
    # Assign to GPUs 2~7 (example)
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file configs/zero3.yaml train_grpo.py --max_tokens 2048
"""

import requests
import io
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate.utils import broadcast_object_list, gather_object
from accelerate.utils import tqdm
import wandb

from reward_fn import reward_funcs_registry

def get_generation_from_vllm(prompts, num_gen=1, temperature=0.8, max_tokens=128, vllm_server_url=None):
    """api call for generation from vLLM"""
    payload = {
        "prompts": prompts,
        "num_gen": num_gen,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    resp = requests.post(f"{vllm_server_url}/generate", json=payload, timeout=500)
    resp.raise_for_status()
    data = resp.json()
                               # <----num_gen---->
    return data["generations"] # [[gen1_1, gen1_2], [gen2_1, gen2_2], ...]

def compute_logprob_from_ref(prompt_and_gen, ref_model_api_url=None):
    """api call for log probability of reference model"""
    prompts, generations = zip(*prompt_and_gen)
    try:
        payload = {
            "prompts": prompts,
            "generations": generations
        }

        resp = requests.post(f"{ref_model_api_url}/logprob", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        return torch.tensor(data["logprobs"])  # [[logp1, logp2], [logp3, logp4], ...]
    except requests.exceptions.RequestException as e:
        print(f"Error requesting ref_model API: {e}")
        return None

def compute_logprob(model, tokenizer, prompt_and_gen):
    """copied from https://github.com/huggingface/trl/blob/249fe97158612839255468892bf74a4d823c1bc6/trl/trainer/grpo_trainer.py#L430"""
    prompts, generations = zip(*prompt_and_gen)
    prompt_input_ids = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True, padding_side="left").to('cuda')
    gen_input_ids = tokenizer.batch_encode_plus(generations, return_tensors="pt", padding=True, padding_side="right").to('cuda')

    prompt_len = prompt_input_ids.input_ids.shape[1]
    input_ids = torch.cat([prompt_input_ids.input_ids, gen_input_ids.input_ids], dim=1).long() # (B, L)
    
    logits = model(input_ids, use_cache=False).logits
    logits = logits[:, :-1, :]  # (B, L-1, V)
    input_ids = input_ids[:, 1:]  # (B, L-1)
    
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)

    per_token_logps = torch.stack(per_token_logps)
    per_token_logps = per_token_logps[:, prompt_len -1 :]

    is_eos = gen_input_ids.input_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=gen_input_ids.input_ids.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=gen_input_ids.input_ids.device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    return per_token_logps, completion_mask

def save_with_accelerate(accelerator, model, output_dir):
    accelerator.deepspeed_plugin.zero3_save_16bit_model = True
    accelerator.deepspeed_plugin.stage3_gather_16bit_weights_on_model_save = True

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    # check if state_dict is a dict has empty tensor
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            output_dir, 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save, 
            state_dict=state_dict
        )

def main(args):
    # 1) Accelerate 초기화
    MODEL_NAME = args.model_name
    VLLM_SERVER_URL = args.vllm_server_url
    REF_MODEL_API_URL = args.ref_model_api_url
    SYSTEM_PROMPT = args.system_prompt

    accelerator = Accelerator()
    local_rank = accelerator.local_process_index

    device = accelerator.device
    per_device_train_batch_size = args.batch_size
    epochs = args.epochs
    num_gen = args.num_gen
    max_tokens = args.max_tokens
    beta = args.beta
    reward_funcs  = [
        reward_funcs_registry["accuracy"],
        reward_funcs_registry["format"],
    ]
    gradient_accumulation_steps = accelerator.deepspeed_plugin.gradient_accumulation_steps
    num_gpus = accelerator.num_processes

    # 2) dataset loading (must have 'problem' and 'solution' fields)
    dataset = load_dataset(args.dataset_name, split="train")
    train_dataloader = DataLoader(dataset, batch_size=per_device_train_batch_size, shuffle=True)

    # 3) model & tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    policy_model.train()
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    # Accelerate prepare
    policy_model, train_dataloader, optimizer = accelerator.prepare(policy_model, train_dataloader, optimizer)

    progress_bar = tqdm(total=len(train_dataloader) * epochs, desc=f"Training")

    # wandb init
    if accelerator.is_main_process:
        wandb.init(
            project="xDAN-RLAIF-GRPO-r1",
            config={
                "learning_rate": args.lr,
                "epochs": epochs,
                "batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_gen": num_gen,
                "max_tokens": max_tokens,
                "beta": beta,
                "model_name": MODEL_NAME,
                "dataset_name": args.dataset_name,
                "system_prompt": SYSTEM_PROMPT,
            },
        )

    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            _metrics = {}
            _metrics["epoch"] = epoch
            _metrics["iteration"] = step

            batch_size = len(batch["problem"])
            prompts : list[str] = batch["problem"]
            prompts = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}] for prompt in prompts]
            prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts] # [batch_size]

            all_prompts : list[str] = gather_object(prompts) # [num_gpus * batch_size]

            generations = [[None] * (num_gen)] * (batch_size * num_gpus)
            if accelerator.is_main_process:
                # len(all_prompts) = 6 (cause of 6 GPUs)
                generation_start_time = time.time()
                generations : list[list[str]] = get_generation_from_vllm(all_prompts, num_gen=num_gen, max_tokens=max_tokens, vllm_server_url=VLLM_SERVER_URL)
                generation_end_time = time.time()
            
            generation_global = broadcast_object_list(generations, from_process=0)
            generations = generation_global[accelerator.process_index * batch_size : accelerator.process_index * batch_size + batch_size] # list[ gen x num_gen ]
            local_generations = [item for sublist in generations for item in sublist] 
            local_prompts = [p for p in prompts for _ in range(num_gen)]
                        
            prompt_and_gen = [(prompt, gen) for prompt, gen in zip(local_prompts, local_generations)]

            with accelerator.accumulate():
                policy_logp_start_time = time.time()
                logp, completion_mask = compute_logprob(policy_model, tokenizer, prompt_and_gen)
                policy_logp_end_time = time.time()
                ref_logp_start_time = time.time()
                ref_logp = compute_logprob_from_ref(prompt_and_gen, ref_model_api_url=REF_MODEL_API_URL)
                ref_logp_end_time = time.time()
                ref_logp = ref_logp.to(accelerator.device)
                per_token_kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
            
                # reward calculation
                rewards_per_func = torch.zeros(len(prompts) * num_gen, len(reward_funcs), device=device)
                
                for i, reward_func in enumerate(reward_funcs):
                    reward_kwargs = {key: [] for key in batch.keys() if key not in ["prompt", "completion"]}

                    reward_kwargs['problem'] = [p for p in batch["problem"] for _ in range(num_gen)]
                    reward_kwargs['solution'] = [sol for sol in batch["solution"] for _ in range(num_gen)]
                    reward_kwargs['completions'] = local_generations
                    
                    rewards = reward_func(**reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(rewards)
                
                rewards = rewards_per_func.sum(dim=1) # [batch_size * num_gen]
                
                # Calculate advantage
                mean_grouped_rewards = rewards.view(-1, num_gen).mean(dim=1)
                std_grouped_rewards = rewards.view(-1, num_gen).std(dim=1)
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_gen, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_gen, dim=0)
                advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

                per_token_loss = torch.exp(logp - logp.detach()) * advantages.unsqueeze(1)
                per_token_loss = -(per_token_loss - beta * per_token_kl)
                loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

                backward_start_time = time.time()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                backward_end_time = time.time()

            # weighy sync 
            sync_start_time = time.time()
            accelerator.wait_for_everyone()
            if (step + 1) % gradient_accumulation_steps == 0:
                full_state_dict = accelerator.get_state_dict(policy_model) # ! IMPORTANT : this takes approx 7 seconds for 7B model

            if accelerator.is_main_process and (step + 1) % gradient_accumulation_steps == 0:
                buffer = io.BytesIO()
                torch.save(full_state_dict, buffer)
                buffer.seek(0)
                try:
                    r = requests.post(f"{VLLM_SERVER_URL}/load_weights", data=buffer.read(), timeout=500)
                    r.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] Failed to load weights to vLLM: {e}")
            sync_end_time = time.time()
            accelerator.wait_for_everyone()

            # for log
            completion_length = accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
            reward_per_func = accelerator.gather_for_metrics(rewards_per_func).mean(0)
            for i, reward_func in enumerate(reward_funcs):
                reward_func_name = reward_func.__name__
                _metrics[f"rewards/{reward_func_name}"] = reward_per_func[i].item()
            _metrics["reward"] = accelerator.gather_for_metrics(rewards).mean().item()
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            _metrics["kl"] = accelerator.gather_for_metrics(mean_kl).mean().item()
            _metrics["completion_length"] = completion_length
            _metrics["loss"] = accelerator.gather_for_metrics(loss).mean().item()
            _metrics["epoch"] = epoch + 1
            _metrics["iteration"] = global_step + 1

            if accelerator.is_main_process:
                wandb.log(_metrics)
                wandb.log({"generation_table":  wandb.Table(columns=["step", "prompt", "generation"], data=[[global_step + 1, local_prompts[0], local_generations[0]]])}, step=global_step + 1)
                progress_bar.update(1)

                print(f"[step : {step}] Generation: {generation_end_time - generation_start_time:.2f} | Policy logp: {policy_logp_end_time - policy_logp_start_time:.2f} | Ref logp: {ref_logp_end_time - ref_logp_start_time:.2f} | Backward: {backward_end_time - backward_start_time:.2f} | Sync: {sync_end_time - sync_start_time:.2f}")

            if (step) % args.save_step == 0:    
                 # saveing every epoch
                accelerator.wait_for_everyone()
                save_with_accelerate(accelerator, policy_model, f"checkpoints/policy_model_{global_step}")
                accelerator.wait_for_everyone()
                print(f"Model saved to checkpoints/policy_model_{global_step}")
            
            global_step += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gen", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04, help="KL divergence weight")
    parser.add_argument("--model_name", type=str, default="Seungyoun/Qwen2.5-7B-Open-R1-Distill")
    parser.add_argument("--dataset_name", type=str, default="AI-MO/NuminaMath-TIR")
    parser.add_argument("--vllm_server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--ref_model_api_url", type=str, default="http://localhost:8001")
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--system_prompt", type=str, default=(
        "A conversation between User and Assistant. The user asks a mathematical question, and the Assistant solves it. "
        "The assistant first thinks about the solution step by step, showing all reasoning clearly. The reasoning "
        "process should be enclosed within <think> </think> tags, and the final answer should be provided within "
        "<answer> </answer> tags with the result in \\boxed{}. For example: "
        "<think>Let's solve this step by step:\n1) First...\n2) Then...\n3) Finally...</think>"
        "<answer>\\boxed{final answer here}</answer>"
    ))
    args = parser.parse_args()


    main(args)