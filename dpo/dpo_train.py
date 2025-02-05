import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import DPOConfig, DPOTrainer
import json


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="/home/zhengqing/glm-4-9b-chat", metadata={"help": "the location of the SFT model name or path"})
    adapter_path: Optional[str] = field(default="/home/zhengqing/KS/dpo_change_0_third/results/final_checkpoint", metadata={"help": "the path to the saved LoRA adapter"})  #The parameters of the first dpo are loaded during the second DPO fine-tuning
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=2, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})
    gradient_checkpointing_use_reentrant: Optional[bool] = field(default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"})
    lora_alpha: Optional[float] = field(default=8, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=150, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=15, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=5, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="/home/zhengqing/KS/dpo_change_0_third/result", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": 'The list of integrations to report the results and logs to.'})
    ignore_bias_buffers: Optional[bool] = field(default=False, metadata={"help": "fix for DDP issues with LM bias/mask buffers"})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

def load_preformatted_dataset(json_file: str) -> Dataset:
    with open(json_file, "r") as f:
        data = json.load(f)
    
        for item in data:
            item["prompt"] = "".join(item["prompt"]) if isinstance(item["prompt"], list) else item["prompt"]
            item["chosen"] = "".join(item["chosen"]) if isinstance(item["chosen"], list) else item["chosen"]
            item["rejected"] = "".join(item["rejected"]) if isinstance(item["rejected"], list) else item["rejected"]
    
            if item["prompt"] is None:
                item["prompt"] = ""
            if item["chosen"] is None:
                item["chosen"] = ""
            if item["rejected"] is None:
                item["rejected"] = ""

    # Convert the loaded JSON data into a dataset
    dataset = Dataset.from_list(data)
    return dataset
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. Load base model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,  # Load the base model
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    # 2. Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        script_args.adapter_path,
        torch_dtype=torch_dtype,
    )

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:   # 9b-chat does not have bos_token, set it, or dpo train will report an error.
            tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
            tokenizer.bos_token_id = tokenizer.eos_token_id

    data_file = "training_data.json"
    train_dataset = load_preformatted_dataset(data_file)
    print(train_dataset)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )
    # # 3. Load evaluation dataset
    evaldata_file = "validation_data.json"
    evaldata_dataset = load_preformatted_dataset(evaldata_file)
    evaldata_dataset = evaldata_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )
    # 4. initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        # eval_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_glm49bchat",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "query_key_value","dense", "dense_h_to_4h"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=evaldata_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)