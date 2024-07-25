import argparse, os

from CoNLL_reader import convert_reviews_to_dict
from CoNLL_reader import read_doc

from transformers import XLNetTokenizer, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def convert_dict_to_set(dict):
    data_set = []
    for i, key in enumerate(dict.keys()):
        item = {'id': i, 'ner_tags': [], 'tokens': []}
        for word_line in dict[key]['word-lines']:
            if len(word_line) > 1:
                if word_line[10].strip() == 'O':
                    item['ner_tags'].append(0)
                elif word_line[10].strip() == 'B-feature':
                    item['ner_tags'].append(1)
                elif word_line[10].strip() == 'I-feature':
                    item['ner_tags'].append(2)
                item['tokens'].append(word_line[2])
        data_set.append(item)
    return data_set

block_size = 128

def group_texts(examples):

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def preprocess_function(data):
    return tokenizer([" ".join(x) for x in data["tokens"]])

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', required=True, help="File containing the raw data")
ap.add_argument('-o', '--output', required=True, help="Output folder to save checkpoints")
ap.add_argument('-b', '--batch-size', required=True, help="Batch size", type=int, default=16)
ap.add_argument('-e', '--epochs', required=True, help="Epochs", type=int, default=10)
ap.add_argument('-m', '--model', required=True, help="Model")

args = vars(ap.parse_args())

file_path = args['file']
output_path = args['output']
batch_size = args['batch_size']
epochs = args['epochs']
model_name = args['model']
#wandb.run.name = model_name

# Step 1: Load data
set_file = read_doc(file_path)
set_dict = convert_reviews_to_dict(set_file)
set = convert_dict_to_set(set_dict)

train_set, eval_set = train_test_split(set, test_size=0.01, random_state=42)

train_set_d = Dataset.from_list(train_set)
eval_set_d = Dataset.from_list(eval_set)
data_set = DatasetDict({'train': train_set_d, 'test': train_set_d})

# Step 2: Tokenize the Corpus
if 'xlnet' in model_name:
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
    print("Causal LM tokenizer: " + model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("MLM tokenizer: " + model_name)

tokenized_reviews = data_set.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=data_set['train'].column_names,
)

dataset = tokenized_reviews.map(group_texts, batched=True, num_proc=4)

if 'xlnet' in model_name:
    tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

if 'xlnet' in model_name:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Causal LM model: " + model_name)
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    print("MLM model: " + model_name)

training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    learning_rate=3e-05,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_dir=os.path.join(output_path, "logs"),
    logging_steps=500,
    #eval_steps=100,
    #max_steps=100,
    #save_steps=100,
    #report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

trainer.train()

import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
