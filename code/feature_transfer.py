import argparse
import re
import json

def read_doc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def convert_features_to_dict(lines):
    i = 0
    features_dict = {}
    found = False
    for line in lines:
        # Skip first line and keep the id
        if i == 0:
            id = re.search(r'id="([^"]+)"', line).group(1)
            app_package = re.search(r'package_name="([^"]+)"', line).group(1)

            # If the feature was never found before, create object in dict
            if id not in features_dict:
                features_dict[id] = {'packages': [], 'word-lines': []}
            # Else, it was already filled, so ignore:
            else:
                found = True
            
            # Add the app reference if it was not there before
            if app_package not in features_dict[id]['packages']:
                features_dict[id]['packages'].append(app_package)

        elif line.strip() == "</doc>":
            i = -2
            found = False

        elif line.strip() != "" and not found:
            features_dict[id]['word-lines'].append(line.split('\t'))

        i += 1
    return features_dict

def match_feature(lines, i, features_conll):
    for feature in features_conll:
        
        # Get lemmatized feature
        lemmatized_feature = []
        for word_line in features_conll[feature]['word-lines']:
            lemmatized_feature.append(word_line[2])
        lemmatized_feature = " ".join(lemmatized_feature)

        # Get lemmatized review
        lemmatized_review = []
        for k in range(0,len(features_conll[feature]['word-lines'])):
            if i+k >= len(lines) or lines[i+k].strip() == '':
                break
            if (len(lines[i+k].split('\t')) < 3):
                print(lines[i+k])
            lemmatized_review.append(lines[i+k].split('\t')[2])
        lemmatized_review = " ".join(lemmatized_review)

        # Check if lemmas match
        if lemmatized_feature == lemmatized_review:
            return feature, len(features_conll[feature]['word-lines'])
        
    return None, 0

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--features-file', required=True, help="Input file containing the set of features (CoNLL format).")
ap.add_argument('-r', '--reviews-file', required=True, help="Input file containing the set of reviews (CoNLL format).")
ap.add_argument('-o', '--output', required=True, help="Output file of annotated reviews (CoNLL format)")

args = vars(ap.parse_args())

features_file_path = args['features_file']
reviews_file_path = args['reviews_file']
output_file_path = args['output']

# Clean output file if it existed
open(output_file_path, 'w').close()

# Read the reviews file
with open(reviews_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Process features
features_file = read_doc(features_file_path)
features_conll = convert_features_to_dict(features_file)

formatted_doc = ""
i = 0
with open(output_file_path, 'a', encoding='utf-8') as file_object:
    for j, line in enumerate(lines):
        if j%10000 == 0:
            print(str(j) + "/" + str(len(lines)) + " processed")

        # Output the first row as found
        if i == 0:
            file_object.write(line)
            id = re.search(r'id="([^"]+)"', line).group(1)
        
        # Output the document end row as found and prepare for iterating
        elif line.strip() == "</doc>":
            file_object.write(line)
            i = -2

        elif line.strip() != "":
            feature, length = match_feature(lines, j, features_conll)

            # If beginning of feature, annotate this (B) and all upcoming lines (I)
            if length > 0:
                line = line.replace('\n','')
                line += "\tB-feature\n" 
                for k in range(1, length):
                    lines[j+k] = lines[j+k].replace('\n','')
                    lines[j+k] += "\tI-feature\n" 

            # If not annotated with either B or I, annotate with O
            elif len(line.split('\t')) == 10:
                line = line.replace('\n','')
                line += "\tO\n"

            file_object.write(line)
        else:
            file_object.write('\n')

        i += 1