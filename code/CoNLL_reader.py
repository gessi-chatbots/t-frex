import re, os

def read_doc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def convert_dict_to_conll(reviews, output_file_path):
    print("Converting dict to CoNLL file...")
    # structure output format
    for review in reviews:
        head = "<doc id=\"" + review + "\" package_name=\"" + reviews[review]['package_name'] + "\" app_name=\"" + reviews[review]['app_name'] + "\" app_category=\"" + str(reviews[review]['app_category']) + "\" google_play_category=\"" + str(reviews[review]['google_play_category']) + "\">"
        if 'unique_id' in reviews[review].keys():
            head = head.replace("package_name=", "unique_id=\"" + reviews[review]['unique_id'] + "\" package_name=")
        tail = "</doc>\n"

        # Output data
        formatted_doc = head + "\n"
        for line in reviews[review]['word-lines']:
            formatted_doc += '\t'.join(line)
        formatted_doc = formatted_doc.strip() + "\n\n" + tail + "\n"

        # Extract the directory path from the file path
        output_directory = os.path.dirname(output_file_path)

        # Create the directory if it does not exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Save data in file
        with open(output_file_path, 'a', encoding='utf-8') as file_object:
            # Append content at the end of file
            file_object.write(formatted_doc)
    print("File saved")

def convert_reviews_to_dict(lines):
    print("Converting reviews to dictionary...\n")
    i = 0
    reviews_dict = {}
    for line in lines:
        # Skip first line and keep the id
        if i == 0:
            id = re.search(r'id="([^"]+)"', line).group(1)
            app_package = re.search(r'package_name="([^"]+)"', line).group(1)
            app_name = re.search(r'app_name="([^"]+)"', line).group(1)
            app_category = re.search(r'app_category="([^"]+)"', line).group(1).replace('[','').replace(']','').replace('\'', '').split(',')
            app_category = [x.strip() for x in app_category]
            gp_category = []
            if re.search(r'google_play_category="([^"]+)"', line) is not None:
                gp_category = re.search(r'google_play_category="([^"]+)"', line).group(1).replace('[','').replace(']','').replace('\'', '').split(',')
                gp_category = [x.strip() for x in gp_category]
            unique_id = None
            if re.search(r'unique_id="([^"]+)"', line) is not None:
                unique_id = re.search(r'unique_id="([^"]+)"', line).group(1)
            # We initialize the review, which will be processed sequentially
            reviews_dict[id] = {'package_name': app_package, 'app_name': app_name, 'app_category': app_category, 'google_play_category': gp_category, 'word-lines': []}
            if unique_id is not None:
                reviews_dict[id]['unique_id'] = unique_id

        elif line.strip() == "</doc>":
            reviews_dict[id]['word-lines'] = reviews_dict[id]['word-lines'][0:len(reviews_dict[id]['word-lines'])-1]
            i = -2

        else:
            reviews_dict[id]['word-lines'].append(line.split('\t'))

        i += 1
    print("Reviews converted\n")
    return reviews_dict