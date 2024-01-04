import stanza
import sys
import argparse
import json
from stanza.utils.conll import CoNLL
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help="Input source file (JSON format).")
ap.add_argument('-o', '--output', required=True, help="Output source file (CoNLL format)")

args = vars(ap.parse_args())

source_file = args['input']
output_file = args['output']

# Clean output file if it existed
open(output_file, 'w').close()

try:
    with open(source_file, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            sys.exit("Input file is not a valid JSON file.")
except FileNotFoundError:
    sys.exit("Specified file not found.")

number_features = 0
for app in data:
    if app['features'] is not None:
        number_features += len(app['features'])

# Build the pipeline using Stanford Stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', use_gpu=True)
features = 0
feature_unique_id = 0
feature_unique_ids = {}

# Process all reviews available in data set (e.g., MApp-KG)
start = time.time()
for app in data:
    if app['features'] is not None and len(app['features']) > 0:
        for feature in app['features']:

            features += 1

            # Assign a unique id to equal features (from source)
            if feature not in feature_unique_ids:
                feature_unique_id += 1
                feature_unique_ids[feature] = feature_unique_id
                feature_id = feature_unique_id
            else:
                feature_id = feature_unique_ids[feature]
            
            # structure output format
            head = "<doc id=\"f_" + str(feature_id) + "\" unique_id=\"af_" + str(features) + "\" package_name=\"" + app['package_name'] + "\" app_name=\"" + app['app_name'] + "\" app_category=\"" + str(app['categories']) + "\">"
            tail = "</doc>"

            # Process review through Stanza NLP pipeline
            doc = nlp(feature)
            conll = CoNLL.convert_dict(doc.to_dict())

            # Output data
            formatted_doc = head + "\n\n"
            for sentence in conll:
                for token in sentence:
                    formatted_doc += "\t".join(token)
                    formatted_doc += "\n"
                formatted_doc += "\n"
            formatted_doc += tail + "\n\n"

            # Save data in file
            with open(output_file, 'a', encoding='utf-8') as file_object:
                # Append content at the end of file
                file_object.write(formatted_doc)

            if features % 1000 == 0:
                print(str(features) + "/" + str(number_features) + " features processed (" + str(time.time() - start) + " seconds)")

end = time.time()

print(str(features) + " features processed")
print(str(end - start) + " seconds")