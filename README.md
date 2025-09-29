# HiLabs Hackathon 2025: Clinical Concept Harmonization Engine

## Introduction
This repository contains the solution for the HiLabs Hackathon 2025, focused on harmonizing clinical concepts. The goal is to build an intelligent engine that maps raw clinical inputs (medications, diagnoses, procedures, and labs) to standardized codes in RxNorm and SNOMED CT.

## Technical Approach
The solution employs a two-stage text-matching algorithm designed to handle variations, abbreviations, and noise in the input data.

### 1. Advanced Text Cleaning
A crucial first step is to preprocess both the input data and the target terminologies (RxNorm and SNOMED CT). The cleaning pipeline involves:
- **Lowercasing and trimming** whitespace.
- **Removing irrelevant information** such as dosages, administration routes, and instructions using regular expressions.
- **Standardizing terminology** by replacing common abbreviations, misspellings, and layman's terms with their correct medical equivalents. This is managed through an external `correction_map.txt` file for easy maintenance.
- **Lemmatization**: Automatically converting words to their base or dictionary form (e.g., "retractions" becomes "retraction"). This makes the matching process more robust by handling pluralization and other word variations without manual hardcoding.
- **Filtering out noise** by removing generic, redundant keywords (e.g., "test," "procedure") that do not contribute to the clinical meaning. This is controlled by the `redundant_keywords.txt` file.

### 2. Two-Stage Matching Algorithm
Once the text is cleaned, a hybrid matching approach is used to find the best possible standardized concept:

- **Stage 1: Candidate Retrieval with TF-IDF**
  - We use the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer to create a numerical representation of the cleaned text from RxNorm and SNOMED CT.
  - For each input description, we calculate its TF-IDF vector and use cosine similarity to efficiently identify the top 20 most likely candidates from the relevant terminology (RxNorm for medications, SNOMED CT for others).
  - This stage acts as a fast and effective filter to narrow down the search space from millions of concepts to a small, manageable set.

- **Stage 2: Precise Matching with Fuzzy Logic**
  - The candidates from Stage 1 are then scored more precisely using a weighted combination of two fuzzy matching scores:
    - **Fuzzy Weighted Ratio (`fuzz.WRatio`)**: This score from `thefuzz` library is robust against differences in word order and minor misspellings.
    - **Word Overlap Score**: A custom score that measures the percentage of words from the input description that are also present in the candidate description. This helps prioritize matches that share key clinical terms.
  - The final score is a weighted average of these two metrics, and the candidate with the highest score is selected as the best match.

This two-stage process combines the speed of TF-IDF for initial filtering with the accuracy of fuzzy logic for fine-grained matching, resulting in a robust and efficient harmonization engine.

## Repository Structure
```
.
├── Target Description Files/
│   ├── rxnorm_all_data.parquet     # RxNorm terminology data
│   └── snomed_all_data.parquet     # SNOMED CT terminology data
├── Test.xlsx                       # Input test data
├── Test_output.xlsx                # Generated output file
├── correction_map.txt              # Knowledge base for term standardization
├── harmonize.py                    # Main Python script for the harmonization engine
├── README.md                       # This file
├── redundant_keywords.txt          # List of non-clinical keywords to remove
├── requirements.txt                # Python dependencies
├── run.sh                          # Shell script to set up and run the project
└── demo.py                         # Standalone script demonstrating the core logic
```

## How to Run the Solution

### Prerequisites
- Python 3

### Instructions
1.  **Clone the repository** (if applicable) and navigate to the project directory.

2.  **Make the run script executable**:
    ```sh
    chmod +x run.sh
    ```

3.  **Execute the script**:
    ```sh
    ./run.sh
    ```

The `run.sh` script will automatically:
- Create a Python virtual environment (`venv`).
- Activate it.
- Install all the required dependencies from `requirements.txt`.
- Run the `harmonize.py` script with the default input (`Test.xlsx`) and output (`Test_output.xlsx`) files.
- Deactivate the virtual environment.

Upon completion, a new file named `Test_output.xlsx` will be created in the directory, containing the original input data with the three new output columns appended.

## Demo Script

For a detailed, step-by-step demonstration of the core logic, you can run the `demo.py` script:

```sh
python3 demo.py
```

This standalone script will execute the entire harmonization process on the `Test.xlsx` file and save the output, providing a clear example of the engine's functionality from start to finish.
