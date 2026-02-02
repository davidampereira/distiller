"""Model distillation with LoRA and 4-bit quantization."""

import gc
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

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


    hf_dataset = Dataset.from_list(dataset)

    divided = hf_dataset.train_test_split(test_size=0.1, seed=42)
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
    training_args = SFTConfig(
        assistant_only_loss = True,
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

    trainer = SFTTrainer(
        model_name,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        peft_config=peft_config
    )

    logger.info("Starting training...")
    trainer.train()

    trainer.save_model(f"{model_save_dir}/adapter")
    logger.info(f"Adapter saved to {model_save_dir}/adapter")

    logger.info(f"Final model saved to {model_save_dir}")
