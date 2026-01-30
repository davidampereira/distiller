"""Model distillation with LoRA and 4-bit quantization."""

import gc
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.config import get
from src.logger import get_logger

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger("finer.distiller")


def distill(model_name: str, dataset: list, model_save_dir: str):
    """Distill knowledge into a smaller model using LoRA fine-tuning.

    Args:
        model_name: HuggingFace model identifier.
        dataset: List of conversation dictionaries.
        model_save_dir: Directory to save the fine-tuned model.
    """
    if not os.path.exists(model_save_dir):
        try:
            os.mkdir(model_save_dir)
            logger.info(f"Created output directory: {model_save_dir}")
        except Exception as e:
            logger.error(f"Couldn't create directory: {e}")
            raise
    elif os.path.isfile(model_save_dir):
        raise ValueError("Path given is not a directory")

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    logger.debug("Cleared GPU memory cache")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def process_data(example):
        full_assistant_content = (
            f"<think>\n{example['chatbot_reasoning']}\n</think>\n{example['chatbot_response']}"
        )
        messages = [  # noqa: F841 - TODO: use this for tokenization
            {"role": "user", "content": example["user_message"]},
            {"role": "assistant", "content": full_assistant_content},
        ]

        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            #return_tensors="pt"
        )

        # TODO: Implement proper tokenization
        return tokenized

    hf_dataset = Dataset.from_list(dataset)
    tokenized = hf_dataset.map(process_data, batched=False)

    divided = tokenized.train_test_split(test_size=0.1, seed=42)
    train = divided["train"]
    test = divided["test"]
    logger.info(f"Dataset split: {len(train)} train, {len(test)} test samples")

    # LoRA configuration from config
    lora_config = get("training.lora", {})
    peft_config = LoraConfig(
        r=lora_config.get("r", 4),
        lora_alpha=lora_config.get("alpha", 8),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments from config
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=get("training.num_epochs", 3),
        per_device_train_batch_size=get("training.batch_size", 1),
        per_device_eval_batch_size=get("training.eval_batch_size", 1),
        gradient_accumulation_steps=get("training.gradient_accumulation_steps", 8),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=get("training.logging_steps", 50),
        learning_rate=get("training.learning_rate", 2e-5),
        fp16=get("training.fp16", True),
        optim="paged_adamw_8bit",
        max_grad_norm=get("training.max_grad_norm", 0.3),
        warmup_ratio=get("training.warmup_ratio", 0.03),
        lr_scheduler_type="cosine",
        push_to_hub=False,
        dataloader_pin_memory=False,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    logger.info("Model prepared for LoRA training")

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=get("training.response_template", "<|think|>"),
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator
    )

    logger.info("Starting training...")
    trainer.train()

    trainer.save_model(f"{model_save_dir}/adapter")
    logger.info(f"Adapter saved to {model_save_dir}/adapter")

    tokenizer.save_pretrained(model_save_dir)

    final_model = trainer.model.merge_and_unload()
    final_model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    logger.info(f"Final model saved to {model_save_dir}")
