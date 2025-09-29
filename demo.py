import pandas as pd
import re
import time
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz

# --- Download NLTK data models ---
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('tokenizers/punkt.zip')
    except LookupError:
        print("Downloading NLTK data models (wordnet, punkt)...")
        nltk.download('wordnet')
        nltk.download('punkt')

# --- 1. Load Knowledge Base from External Files ---
def load_correction_map(filepath="correction_map.txt"):
    correction_map = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    correction_map[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. The cleaner will run without a correction map.")
    return correction_map

def load_redundant_keywords(filepath="redundant_keywords.txt"):
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. The cleaner will run without removing keywords.")
        return []

# --- 2. Advanced Cleaning Pipeline ---
def create_cleaner(correction_map, redundant_keywords):
    lemmatizer = WordNetLemmatizer()

    def clean_and_standardize(text):
        if not isinstance(text, str): return ""
        text = text.lower().strip()
        
        text = re.sub(r'sig:.*', '', text)
        text = re.sub(r'take \d+(\s*\(.*\))? tablet\(s\)?', '', text)
        text = re.sub(r'\d+(\.\d+)?\s*(mg|ml|mcg|unit|units|g|gram|grams)\b', '', text)
        text = re.sub(r'\b(by|via|every|with|as needed|at bedtime|oral|route|injection|topically)\b', '', text)
        
        for key, value in sorted(correction_map.items(), key=lambda item: len(item[0]), reverse=True):
            text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)
            
        for word in redundant_keywords:
            text = re.sub(r'\b' + word + r'\b', '', text)
            
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
        text = " ".join(lemmatized_words)

        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return clean_and_standardize

# --- 3. Two-Stage Harmonization ---
def find_best_match(description, entity_type, snomed_data, rxnorm_data, snomed_tfidf, rxnorm_tfidf, snomed_vectorizer, rxnorm_vectorizer):
    is_snomed = entity_type in ['Procedure', 'Lab', 'Diagnosis']
    data, tfidf_matrix, vectorizer, system = (snomed_data, snomed_tfidf, snomed_vectorizer, 'SNOMEDCT_US') if is_snomed else (rxnorm_data, rxnorm_tfidf, rxnorm_vectorizer, 'RXNORM')
    
    if not description.strip():
        return {'System': system, 'Code': 'NO_INPUT', 'Description': 'Original text was empty after cleaning'}

    desc_tfidf = vectorizer.transform([description])
    similarities = cosine_similarity(desc_tfidf, tfidf_matrix).flatten()
    candidate_indices = similarities.argsort()[-20:][::-1]
    
    if len(candidate_indices) == 0 or similarities[candidate_indices[0]] < 0.1:
        return {'System': system, 'Code': 'NOT_FOUND', 'Description': 'No suitable match found'}

    best_final_score, best_match_index = -1, -1
    input_words = set(description.split())
    
    for idx in candidate_indices:
        candidate_str = data.iloc[idx]['Cleaned_STR']
        wratio_score = fuzz.WRatio(description, candidate_str)
        candidate_words = set(candidate_str.split())
        overlap_score = (len(input_words.intersection(candidate_words)) / len(input_words)) * 100 if input_words else 0
        final_score = (wratio_score * 0.4) + (overlap_score * 0.6)
        
        if final_score > best_final_score:
            best_final_score, best_match_index = final_score, idx

    return {'System': system, 'Code': str(data.iloc[best_match_index]['CODE']), 'Description': data.iloc[best_match_index]['STR']}

# --- Main Execution Block ---
def main():
    # Download NLTK data
    download_nltk_data()

    # --- Load Data and Initialize Cleaner ---
    try:
        test_df = pd.read_excel('Test.xlsx')
        snomed_df = pd.read_parquet('Target Description Files/snomed_all_data.parquet')
        rxnorm_df = pd.read_parquet('Target Description Files/rxnorm_all_data.parquet')
        print("All data files loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all required data files are in the directory.")
        return

    # Load the knowledge base and create the cleaning function
    correction_map = load_correction_map()
    redundant_keywords = load_redundant_keywords()
    cleaner = create_cleaner(correction_map, redundant_keywords)

    print("Applying cleaning pipeline to all text data...")
    test_df['Cleaned_Description'] = test_df['Input Entity Description'].apply(cleaner)
    snomed_df['Cleaned_STR'] = snomed_df['STR'].apply(cleaner)
    rxnorm_df['Cleaned_STR'] = rxnorm_df['STR'].apply(cleaner)

    # --- TF-IDF Preparation ---
    print("Preparing TF-IDF matrices...")
    snomed_vectorizer = TfidfVectorizer(stop_words='english')
    snomed_tfidf = snomed_vectorizer.fit_transform(snomed_df['Cleaned_STR'].fillna(''))
    rxnorm_vectorizer = TfidfVectorizer(stop_words='english')
    rxnorm_tfidf = rxnorm_vectorizer.fit_transform(rxnorm_df['Cleaned_STR'].fillna(''))

    # --- Harmonization Process ---
    print("Running harmonization process...")
    start_time = time.time()
    results = []
    for index, row in test_df.iterrows():
        match = find_best_match(row['Cleaned_Description'], row['Entity Type'], snomed_df, rxnorm_df, snomed_tfidf, rxnorm_tfidf, snomed_vectorizer, rxnorm_vectorizer)
        results.append(match)
    
    print(f"Harmonization finished in {time.time() - start_time:.2f} seconds.")

    # --- Final Output ---
    results_df = pd.DataFrame(results)
    test_df['Output Coding System'] = results_df['System']
    test_df['Output Target Code'] = results_df['Code']
    test_df['Output Target Description'] = results_df['Description']
    
    test_df = test_df.drop(columns=['Cleaned_Description'])
    
    test_df.to_excel('Test_output.xlsx', index=False)
    print(f"Output successfully saved to Test_output.xlsx")

if __name__ == '__main__':
    main()

