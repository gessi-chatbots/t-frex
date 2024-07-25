import argparse
import os
import json
import random
import numpy as np

from CoNLL_reader import convert_dict_to_conll, convert_reviews_to_dict, read_doc
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Categories
categories = ['PRODUCTIVITY', 'COMMUNICATION', 'TOOLS', 'SOCIAL', 'HEALTH_AND_FITNESS',
              'PERSONALIZATION', 'TRAVEL_AND_LOCAL', 'MAPS_AND_NAVIGATION', 'LIFESTYLE', 'WEATHER']

# Bins
bins = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9']

# Instances
data_splits = [0.125, 0.25, 0.50, 0.75]

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument('-if', '--input-folder', required=True, help="Folder containing train and tests sets")
ap.add_argument('-of', '--output-folder', required=True, help="Folder where checkpoints and results will be saved")
ap.add_argument('-sf', '--sub-folders', required=True, help="Sub-folders (i.e., none, category, bin)")

args = vars(ap.parse_args())
input_folder = args['input_folder']
output_folder = args['output_folder']
sf = args['sub_folders']

# Define the iteration of subfolders
if sf == 'category':
    subfolders = categories
elif sf == 'bin':
    subfolders = bins
else:
    subfolders = ['']

# Specify the path to your JSON file
json_file_path = os.path.join("data", "feature-reviews.json")
metadata = []

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_similarity(X):
    similarity_matrix = cosine_similarity(X)
    np.fill_diagonal(similarity_matrix, 0)  # Set diagonal elements to 0 to avoid self-similarity
    return similarity_matrix

for data_split in data_splits:
    
    split_metadata = []

    # Open the JSON file for reading
    with open(json_file_path, "r") as file:
        # Load the JSON data into a Python dictionary
        feature_reviews = json.load(file)
        
        for subfolder in subfolders:
            
            train_set_dict_new = {}
            new_dict = {}
                
            print("Starting instance selection process for " + subfolder)

            # Process reviews
            print("Reading docs...")
            train_set_file = read_doc(os.path.join(input_folder, subfolder, "train-set.txt"))

            print("Done.\nFormatting docs...")
            train_set_dict = convert_reviews_to_dict(train_set_file)
            
            for key, value in feature_reviews.items():
                # Calculate the size of the array
                # Iterate through train_set_dict and count occurrences of key
                count = 0
                for instance_key in train_set_dict:
                    if instance_key in value:
                        count += 1
                # Initialize the 'current' property to 0
                current = 0
                # Create a dictionary entry with the key and the desired properties
                new_dict[key] = {"total": count, "current": current}
            
            # Generate BERT embeddings for the texts
            concatenated_texts = [' '.join(subarray[1] for subarray in review['word-lines'] if len(subarray) > 1) for review in train_set_dict.values()]
            X_train_bert = get_bert_embeddings(concatenated_texts).numpy()

            # Calculate similarity matrix between instances
            similarity_matrix = calculate_similarity(X_train_bert)

            # Calculate average similarity for each instance
            average_similarity = np.mean(similarity_matrix, axis=1)

            # Sort keys based on average similarity (lower similarity means higher priority)
            sorted_keys = [key for _, key in sorted(zip(average_similarity, train_set_dict.keys()))]

            # Filter instances
            for review_id in sorted_keys:
                review = train_set_dict[review_id]
                for feature, reviews in feature_reviews.items():
                    if review_id in reviews:
                        if new_dict[feature]['current'] < new_dict[feature]['total']*data_split:
                            train_set_dict_new[review_id] = review
                            new_dict[feature]['current'] += 1
                            
            split_metadata.append((len(train_set_dict_new)))
                                        
            convert_dict_to_conll(train_set_dict_new, os.path.join(output_folder, subfolder, "train-set_" + str(data_split) + ".txt"))
            
    metadata.append(split_metadata)
print(metadata)
