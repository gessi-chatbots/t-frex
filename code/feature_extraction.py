import argparse
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from tqdm import tqdm

def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Feature extraction from reviews")
    parser.add_argument("input_file", help="Path to the CSV file containing review data")
    parser.add_argument("output_file", help="Path to save the processed data")
    parser.add_argument("--model_id", default="quim-motger/t-frex-bert-base-uncased", help="Hugging Face model ID for feature extraction")
    args = parser.parse_args()

    reviews = read_csv_file(args.input_file)
    print(f"Loaded {len(reviews)} reviews from {args.input_file}")

    # Load the Hugging Face model for feature extraction
    model = AutoModelForTokenClassification.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    feature_extractor = pipeline("ner", model=model, tokenizer=tokenizer)
    print(f"Loaded feature extraction model: {args.model_id}")
    
    # Initialize counters and progress tracking
    total_reviews = len(reviews)
    batch_size = 10
    print(f"Starting feature extraction for {total_reviews} reviews...")
    
    # Create a progress bar
    progress_bar = tqdm(total=total_reviews, desc="Extracting features", unit="review")
    
    for i in range(0, total_reviews, batch_size):
        batch = reviews.iloc[i:i+batch_size]
        
        for idx, review in batch.iterrows():
            review_text = review['ReviewText']
            features = feature_extractor(review_text)
            
            # Process the extracted features
            processed_features = []
            current_feature = ""
            for feature in features:
                if feature['entity'] == 'B-feature':
                    if current_feature:
                        processed_features.append(current_feature.strip())
                    current_feature = feature['word']
                elif feature['entity'] == 'I-feature':
                    current_feature += " " + feature['word']
            
            if current_feature:
                processed_features.append(current_feature.strip())
            
            # Update the 'FeatureLabel' column
            reviews.at[idx, 'FeatureLabel'] = '; '.join(processed_features)
        
        # Update the progress bar
        progress_bar.update(len(batch))
    
    # Close the progress bar
    progress_bar.close()

    # Save the updated DataFrame to the output file
    reviews.to_csv(args.output_file, index=False)
    print(f"Updated dataset with extracted features saved to {args.output_file}")

if __name__ == "__main__":
    main()
