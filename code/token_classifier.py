import argparse, csv
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from tqdm import tqdm
from CoNLL_reader import convert_reviews_to_dict
from CoNLL_reader import read_doc

label_list = ['O','B-feature','I-feature']
id2label = {0: 'O', 1: 'B-feature', 2: 'I-feature'}
label2id = {'O': 0, 'B-feature': 1, 'I-feature': 2}

parser = argparse.ArgumentParser(description="Token classifier")
parser.add_argument("--checkpoint", required=True, help="Path or name of the fine-tuned model checkpoint")
args = parser.parse_args()

checkpoint = args.checkpoint

# Load the model
model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=3, id2label=id2label, label2id=label2id)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Initialize the NER pipeline with your fine-tuned model
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Text for token classification
while 1:
    text = input("Enter the text for classification: ")
    ner_results = ner_pipeline(text)

    features = []
    current_feature = ""

    for feature in ner_results:
        start = feature['start']
        end = feature['end']

        # Hot fix to get the whole word
        blank_space = False
        while start > 0 and not blank_space and not text[start] == ' ':
            if text[start-1] != ' ':
                start -= 1
            else:
                blank_space = True

        blank_space = False
        while end < len(text)-1 and not blank_space and not text[end] == ' ':
            if text[end+1] != ' ':
                end += 1
            else:
                blank_space = True

        f = text[start:end+1]

        if feature['entity'] == 'B-feature':
            #If we were processing a feature, it is saved
            if len(current_feature) > 0:
                features.append(current_feature.strip().lower())
                current_feature = ""
            current_feature += f + " "

        elif feature['entity'] == 'I-feature':
            # hot fix to make sure a feature does not have the same word twice
            if f not in current_feature:
                current_feature += f + " "

        elif feature['entity'] == 'O':
            #If we were processing a feature, it is saved
            if len(current_feature) > 0:
                features.append(current_feature.strip().lower())
                current_feature = ""

    if len(current_feature) > 0:
        features.append(current_feature.strip().lower())

    print("\nToken prediction:")
    print(str(ner_results) + "\n")
    print("Feature extraction: ")
    print(str(set(features)) + "\n")