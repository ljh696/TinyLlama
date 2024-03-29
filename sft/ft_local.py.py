from datasets import load_dataset, load_from_disk
from transformers import (AutoModelForSeq2SeqLM, 
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from peft import LoraConfig, TaskType, get_peft_model
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(1)
model_name_or_path = "google-t5/t5-small"
IGNORE_TOKEN_ID = -100
data_path = "sft/tokenized_data/"

dataset = {}
for split in ["train","test"]:
    dataset[split] = load_from_disk(data_path + split)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    padding_side="right",
    use_fast=True, # Fast tokenizer giving issues.
    trust_remote_code=True,
)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path,
    device_map="cuda:1"
)

lora_config = LoraConfig(
    r=1,  #rank
    lora_alpha=32,  #lora scaling alpha
    target_modules=['q','v'],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

data_collactor = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=IGNORE_TOKEN_ID,
    pad_to_multiple_of=8,
)
output_dir="sft/out"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="no"
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    data_collator = data_collactor,
    train_dataset = dataset['train']
)
model.config.use_cache = False

trainer.train()
