from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
from datasets import Dataset
import evaluate
import os

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(data):
        return tokenizer(data, padding="max_length", truncation=True)

    # tokenized = dataset.map(tokenize, batched=True)
    tokenized = [tokenize(i) for i in dataset]

    def train_test(data):
        length = len(data)
        ninety = int(0.9 * length)
        train = data[:ninety]
        test = data[ninety:]
        return train, test

    train, test = train_test(tokenized)

    trainingArgs = TrainingArguments(
        output_dir=modelSaveDir,
        eval_strategy="epoch",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=trainingArgs,
        # train_dataset=train,
        # eval_dataset = test
        train_dataset = Dataset.from_list(train),
        eval_dataset = Dataset.from_list(test)
    )

    trainer.train()
    trainer.save_model(modelSaveDir)
    tokenizer.save_pretrained(modelSaveDir)

    final_model = trainer.model.merge_and_unload()
    final_model.save_pretrained(modelSaveDir)
    tokenizer.save_pretrained(modelSaveDir)
