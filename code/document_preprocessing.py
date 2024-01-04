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

# Clean output file if exists
open(output_file, 'w').close()

try:
    with open(source_file, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            sys.exit("Input file is not a valid JSON file.")
except FileNotFoundError:
    sys.exit("Specified file not found.")

number_reviews = 0
for app in data:
        number_reviews += len(app['reviews'])

# Build the pipeline using Stanford Stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', use_gpu=True)
reviews = 0

# Process all reviews available in data set
start = time.time()
for app in data:
    for review in app['reviews']:
        
        if review['review'] is not None:

            # structure output format
            head = "<doc id=\"" + review['reviewId'] + "\" package_name=\"" + app['package_name'] + "\" app_name=\"" + app['app_name'] + "\" google_play_category=\"" + str([app['categoryId']]) + "\">"
            tail = "</doc>"

            reviews += 1

            # Process review through Stanza NLP pipeline
            doc = nlp(review['review'])
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

            if reviews % 1000 == 0:
                print(str(reviews) + "/" + str(number_reviews) + " reviews processed (" + str(time.time() - start) + " seconds)")
 
end = time.time()

print(str(reviews) + " reviews processed")
print(str(end - start) + " seconds")