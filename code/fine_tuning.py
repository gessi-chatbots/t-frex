import argparse, torch, evaluate
import numpy as np
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, RobertaTokenizer
from datasets import Dataset, DatasetDict
from CoNLL_reader import convert_reviews_to_dict
from CoNLL_reader import read_doc

# Method to compute precision, recall, f1 and accuracy at token-level
def compute_metrics_token_level(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": None,
    }

# Method to compute precision, recall, f1 and accuracy at feature-level
def compute_metrics_feature_level(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i,sentence in enumerate(true_predictions):

        feature_in_progress = 0

        for j, token in enumerate(sentence):

            if token == 'B-feature':
                feature_in_progress = 1
                if true_labels[i][j] != 'B-feature': 
                    false_positives += 1
                    feature_in_progress = 0
            if token == 'I-feature':
                if true_labels[i][j] != 'I-feature' and feature_in_progress == 1:
                    false_positives += 1
                    feature_in_progress = 0
                if feature_in_progress == 0:
                    false_positives += 1
            if token == 'O':
                if true_labels[i][j] == 'O' and feature_in_progress == 1:
                    true_positives += 1
                feature_in_progress = 0

        if feature_in_progress == 1:
            true_positives += 1

    for i, sentence in enumerate(true_labels):
        feature_in_progress = 0

        for j, token in enumerate(sentence):

            if token == 'B-feature':
                feature_in_progress = 1
                if true_predictions[i][j] != 'B-feature': 
                    false_negatives += 1
                    feature_in_progress = 0
            if token == 'I-feature':
                if true_predictions[i][j] != 'I-feature' and feature_in_progress == 1:
                    false_negatives += 1
                    feature_in_progress = 0
            if token == 'O':
                if feature_in_progress and true_predictions[i][j] != 'O':
                    false_negatives += 1
                feature_in_progress = 0

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": None
    }

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

def tokenize_and_align_labels(data_set):
    tokenized_inputs = tokenizer(data_set["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(data_set[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Initialize structures
seqeval = evaluate.load("seqeval")

label_list = ['O','B-feature','I-feature']
id2label = {0: 'O', 1: 'B-feature', 2: 'I-feature'}
label2id = {'O': 0, 'B-feature': 1, 'I-feature': 2}

# List of valid models
models = ['bert-base-uncased',
          'bert-large-uncased',
          'roberta-base',
          'roberta-large',
          'xlnet-base-cased',
          'xlnet-large-cased']

# Categories 
categories = ['PRODUCTIVITY',
              'COMMUNICATION',
              'TOOLS',
              'SOCIAL',
              'HEALTH_AND_FITNESS',
              'PERSONALIZATION',
              'TRAVEL_AND_LOCAL',
              'MAPS_AND_NAVIGATION',
              'LIFESTYLE',
              'WEATHER']

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help="Model to fine-tune from the list: " + str(models))
ap.add_argument('-if', '--input-folder', required=True, help="Folder containing train and tests sets")
ap.add_argument('-of', '--output-folder', required=True, help="Folder where checkpoints and results will be saved")
ap.add_argument('-e', '--eval', required=True, help="Evaluation strategy (i.e, token-level, feature-level)")

ap.add_argument('-ep', '--epochs', required=True, help="Number of epochs for training parameters")
ap.add_argument('-lr', '--learning-rate', required=True, help="Learning rate for training parameters")
ap.add_argument('-bs', '--batch-size', required=True, help="Batch size for training parameters")

args = vars(ap.parse_args())
model_name = args['model']
input_folder = args['input_folder']
output_folder = args['output_folder']
eval = args['eval']

epochs = args['epochs']
learning_r = args['learning_rate']
batch_size = args['batch_size']

# Validate model is expected value
if model_name not in models:
    raise Exception("Invalid 'model' value. It should be one of the following: " + str(models) + ".") 

if eval == 'feature_level':
    compute_metrics_func = compute_metrics_feature_level
elif eval == 'token_level':
    compute_metrics_func = compute_metrics_token_level
else:
    raise ValueError("Invalid 'eval' value. It should be 'feature_level' or 'token_level'.")

# Load tokenizer. Special parameters required for RoBERTa-based models
if 'roberta' in model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
 
torch.cuda.empty_cache()

# Process reviews
print("Reading docs...")
train_set_file = read_doc(input_folder + "/train-set.txt")
test_set_file = read_doc(input_folder + "/test-set.txt")

print("Done.\nFormatting docs...")
train_set_dict = convert_reviews_to_dict(train_set_file)
test_set_dict = convert_reviews_to_dict(test_set_file)

# Format data for model training
train_set = convert_dict_to_set(train_set_dict)
test_set = convert_dict_to_set(test_set_dict)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=3, id2label=id2label, label2id=label2id
)

# Tokenized data set 
train_set_d = Dataset.from_list(train_set)
test_set_d = Dataset.from_list(test_set)
data_set = DatasetDict({'train': train_set_d, 'test': test_set_d})
data_set = data_set.map(tokenize_and_align_labels, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir= output_folder + "/model",
    learning_rate=learning_r,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Train the model
print("Fine-tuning " + model_name)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_set['train'],
    eval_dataset=data_set['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_func,
)

print("Starting fine-tuning...")
trainer.train()

print("Done. Evaluating...")
result = trainer.evaluate(eval_dataset=data_set['test'])
