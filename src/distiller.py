from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from datasets import Dataset
import evaluate
import os
import torch
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
metric = evaluate.load("accuracy")

def distill(model_name, dataset, modelSaveDir):

    if os.path.exists(modelSaveDir) == False:
        try:
            os.mkdir(modelSaveDir)
        except Exception as e:
            print("Couldn't create given directory")
            print(e)
            exit()
    elif os.path.isfile(modelSaveDir):
        raise Exception("Path given is not directory")

    # Clear any existing GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
    )

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
        full_assistant_content = f"<think>\n{example['chatbot_reasoning']}\n</think>\n{example['chatbot_response']}"

        messages = [
            {"role": "user", "content": example["user_message"]},
            {"role": "assistant", "content": full_assistant_content}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_dict=True
        )

        inputs["labels"] = inputs["input_ids"]

        return inputs

    hf_dataset = Dataset.from_list(dataset)
    tokenized = hf_dataset.map(process_data, batched=False)

    divided = tokenized.train_test_split(test_size=0.1, seed=42)
    train = divided["train"]
    test = divided["test"]

    # LoRA configuration - reduced rank for memory efficiency
    peft_config = LoraConfig(
        r=4,  # Reduced rank for lower memory usage
        lora_alpha=8,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
        target_modules=["q_proj", "v_proj"],  # Only essential attention layers
        bias="none",  # No bias term
        task_type="CAUSAL_LM"  # Task type
    )

    trainingArgs = TrainingArguments(
        output_dir=modelSaveDir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Increased for memory efficiency
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        fp16=True,
        optim="paged_adamw_8bit",  # Use 8-bit optimizer for less memory
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        dataloader_pin_memory=False,  # Reduce memory pressure
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=trainingArgs,
        train_dataset = train,
        eval_dataset = test,
    )

    trainer.train()
    trainer.save_model(f"modelSaveDir/adapter")
    print("Model Saved")
    tokenizer.save_pretrained(modelSaveDir)

    final_model = trainer.model.merge_and_unload()
    final_model.save_pretrained(modelSaveDir)
    tokenizer.save_pretrained(modelSaveDir)
