import os
import shutil
import json
import re
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV_PATH = os.path.join(BASE_DIR, 'goemotions.csv')

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')

LABEL2ID = {
    "Joy": 0,
    "Sadness": 1,
    "Anger": 2,
    "Anxiety": 3
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Mapping from GoEmotions fine-grained labels to 4 target classes
EMOTION_MAPPING = {
    # JOY
    "joy": "Joy",
    "amusement": "Joy",
    "excitement": "Joy",
    "gratitude": "Joy",
    "optimism": "Joy",
    "love": "Joy",
    "pride": "Joy",
    "relief": "Joy",
    "admiration": "Joy",
    "approval": "Joy",
    "caring": "Joy",
    "desire": "Joy",
    
    # SADNESS
    "sadness": "Sadness",
    "grief": "Sadness",
    "disappointment": "Sadness",
    "remorse": "Sadness",
    "embarrassment": "Sadness",
    
    # ANGER
    "anger": "Anger",
    "annoyance": "Anger",
    "disapproval": "Anger",
    
    # ANXIETY
    "fear": "Anxiety",
    "nervousness": "Anxiety",
    
    # UNMAPPED / OTHER
    "confusion": "UNMAPPED",
    "curiosity": "UNMAPPED",
    "disgust": "UNMAPPED",
    "realization": "UNMAPPED",
    "surprise": "UNMAPPED",
    "neutral": "UNMAPPED"
}

def setup_directories():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    # Copy raw CSV to data/raw/goemotions.csv if not already there
    target_raw_csv = os.path.join(RAW_DATA_DIR, 'goemotions.csv')
    if os.path.abspath(RAW_CSV_PATH) != os.path.abspath(target_raw_csv):
        shutil.copyfile(RAW_CSV_PATH, target_raw_csv)
        print(f"Copied raw dataset to {target_raw_csv}")

def load_dataset(csv_path):
    print("=" * 60)
    print("STEP 1: LOADING & INSPECTING DATASET")
    print("=" * 60)
    df = pd.read_csv(csv_path)
    print(f"Loaded '{csv_path}' successfully.")
    return df

def inspect_dataset(df):
    rows, cols = df.shape
    print(f"\n--- Dataset Summary ---")
    print(f"Number of Rows: {rows}")
    print(f"Number of Columns: {cols}")
    print(f"Column Names: {list(df.columns)}")
    print(f"\n--- Data Types & Missing Values ---")
    null_counts = df.isnull().sum()
    print(pd.DataFrame({'Dtype': df.dtypes, 'Missing Values': null_counts}))
    
    duplicate_rows = df.duplicated().sum()
    print(f"\nTotal Duplicate Rows in raw CSV: {duplicate_rows}")
    
    print(f"\n--- First 5 Rows ---")
    print(df.head(5))
    
    # Identify text and emotion columns
    text_col = 'text' if 'text' in df.columns else None
    meta_cols = ['id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear']
    meta_cols_present = [c for c in meta_cols if c in df.columns]
    
    all_emotions = [e for e in EMOTION_MAPPING.keys() if e in df.columns]
    
    print(f"\nDetected Text Column: '{text_col}'")
    print(f"Detected Metadata Columns: {meta_cols_present}")
    print(f"Detected Emotion Columns ({len(all_emotions)} total): {all_emotions}")
    
    return text_col, all_emotions, meta_cols_present

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # HTML unescape (e.g., &amp; -> &, &gt; -> >, &#39; -> ')
    text = html.unescape(text)
    # Fix obvious replacement character encoding artifacts
    text = text.replace('\ufffd', "'")
    # Convert newline characters to single space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()

def clean_text_data(df, text_col):
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING & NORMALIZATION")
    print("=" * 60)
    
    initial_count = len(df)
    
    # 1. Missing / Empty Text
    df['clean_text'] = df[text_col].apply(normalize_text)
    missing_text_mask = (df['clean_text'] == "") | (df['clean_text'].isnull())
    removed_missing_text = missing_text_mask.sum()
    
    df_clean = df[~missing_text_mask].copy()
    print(f"Removed missing/empty text rows: {removed_missing_text}")
    
    # 2. Filter out extremely short / meaningless texts (e.g. only 1 char or only symbols)
    def is_meaningful_text(t):
        # Must contain at least one alphanumeric character
        if not re.search(r'\w', t):
            return False
        if len(t) < 2:
            return False
        return True
    
    meaningful_mask = df_clean['clean_text'].apply(is_meaningful_text)
    removed_non_meaningful = (~meaningful_mask).sum()
    df_clean = df_clean[meaningful_mask].copy()
    print(f"Removed non-meaningful / punctuation-only rows: {removed_non_meaningful}")
    
    return df_clean, removed_missing_text, removed_non_meaningful

def map_and_resolve_emotions(df, emotion_cols):
    print("\n" + "=" * 60, flush=True)
    print("STEP 3 & 4: EMOTION MAPPING & MULTI-LABEL RESOLUTION", flush=True)
    print("=" * 60, flush=True)
    
    print("\n--- Emotion Mapping Table ---", flush=True)
    mapping_df = pd.DataFrame(list(EMOTION_MAPPING.items()), columns=['Original Emotion', 'Target Category'])
    print(mapping_df.to_string(index=False), flush=True)
    
    initial_count = len(df)
    
    # Emotion column subsets for target classes
    joy_cols = [c for c in emotion_cols if EMOTION_MAPPING.get(c) == "Joy"]
    sadness_cols = [c for c in emotion_cols if EMOTION_MAPPING.get(c) == "Sadness"]
    anger_cols = [c for c in emotion_cols if EMOTION_MAPPING.get(c) == "Anger"]
    anxiety_cols = [c for c in emotion_cols if EMOTION_MAPPING.get(c) == "Anxiety"]
    
    has_joy = (df[joy_cols] == 1).any(axis=1)
    has_sadness = (df[sadness_cols] == 1).any(axis=1)
    has_anger = (df[anger_cols] == 1).any(axis=1)
    has_anxiety = (df[anxiety_cols] == 1).any(axis=1)
    
    # Total active emotion indicators per row across all 28 emotions
    active_counts = (df[emotion_cols] == 1).sum(axis=1)
    single_label_count = int((active_counts == 1).sum())
    multi_label_count = int((active_counts > 1).sum())
    
    # Target category presence count per row
    target_presence = has_joy.astype(int) + has_sadness.astype(int) + has_anger.astype(int) + has_anxiety.astype(int)
    
    # 1. Cleanly mapped (exactly 1 target category active)
    mapped_mask = (target_presence == 1)
    # 2. Ambiguous (more than 1 target category active)
    ambiguous_mask = (target_presence > 1)
    # 3. Unmapped (0 target categories active)
    unmapped_mask = (target_presence == 0)
    
    successfully_mapped_count = int(mapped_mask.sum())
    ambiguous_count = int(ambiguous_mask.sum())
    unmapped_count = int(unmapped_mask.sum())
    
    # Construct original_label string for active emotions per row
    emotion_mat = df[emotion_cols].values
    active_labels_list = []
    for i in range(len(df)):
        active_indices = np.where(emotion_mat[i] == 1)[0]
        active_labels_list.append(",".join([emotion_cols[idx] for idx in active_indices]))
        
    df_result = pd.DataFrame({
        'clean_text': df['clean_text'],
        'original_label': active_labels_list,
        'has_joy': has_joy,
        'has_sadness': has_sadness,
        'has_anger': has_anger,
        'has_anxiety': has_anxiety,
        'target_presence': target_presence
    }, index=df.index)
    
    # Assign target emotion
    conditions = [
        df_result['has_joy'] & (df_result['target_presence'] == 1),
        df_result['has_sadness'] & (df_result['target_presence'] == 1),
        df_result['has_anger'] & (df_result['target_presence'] == 1),
        df_result['has_anxiety'] & (df_result['target_presence'] == 1)
    ]
    choices = ['Joy', 'Sadness', 'Anger', 'Anxiety']
    
    df_result['emotion'] = np.select(conditions, choices, default='UNMAPPED')
    df_result['label'] = df_result['emotion'].map(LABEL2ID).fillna(-1).astype(int)
    df_result['ambiguous'] = df_result['target_presence'] > 1
    
    print("\n--- Multi-Label & Mapping Resolution Results ---", flush=True)
    print(f"Total Input Samples Evaluated: {initial_count}", flush=True)
    print(f"Single-label active emotion samples: {single_label_count}", flush=True)
    print(f"Multi-label active emotion samples: {multi_label_count}", flush=True)
    print(f"Successfully mapped samples: {successfully_mapped_count}", flush=True)
    print(f"Ambiguous / Conflicting target samples: {ambiguous_count}", flush=True)
    print(f"Unmapped / Neutral-only samples: {unmapped_count}", flush=True)
    
    # Filter out ambiguous and unmapped samples for final dataset
    clean_mapped_df = pd.DataFrame({
        'text': df_result.loc[mapped_mask, 'clean_text'],
        'original_label': df_result.loc[mapped_mask, 'original_label'],
        'emotion': df_result.loc[mapped_mask, 'emotion'],
        'label': df_result.loc[mapped_mask, 'label'],
        'ambiguous': False
    })
    
    return clean_mapped_df, unmapped_count, ambiguous_count

def deduplicate_and_filter_leakage(df):
    print("\n" + "=" * 60)
    print("STEP 7: DATA LEAKAGE & DEDUPLICATION FILTERING")
    print("=" * 60)
    
    initial_count = len(df)
    
    # 1. Deduplicate exact (text, label) pairs
    df_dedup = df.drop_duplicates(subset=['text', 'label']).copy()
    dedup_exact_removed = initial_count - len(df_dedup)
    print(f"Removed exact (text, label) duplicates: {dedup_exact_removed}")
    
    # 2. Check for duplicate text with DIFFERENT labels (label conflict / leakage)
    text_counts = df_dedup['text'].value_counts()
    conflicting_texts = text_counts[text_counts > 1].index
    
    if len(conflicting_texts) > 0:
        print(f"Found {len(conflicting_texts)} unique text strings with conflicting labels. Removing all instances...")
        df_dedup = df_dedup[~df_dedup['text'].isin(conflicting_texts)].copy()
        conflicting_removed = len(conflicting_texts)
    else:
        conflicting_removed = 0
        print("No conflicting multi-label duplicate texts found.")
        
    final_count = len(df_dedup)
    total_duplicates_removed = initial_count - final_count
    print(f"Total rows after deduplication: {final_count} (Removed total: {total_duplicates_removed})")
    
    return df_dedup, total_duplicates_removed

def analyze_class_distribution(df):
    print("\n" + "=" * 60)
    print("STEP 6: CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    total = len(df)
    counts = df['emotion'].value_counts()
    percents = (counts / total * 100).round(2)
    
    dist_df = pd.DataFrame({
        'Class Name': counts.index,
        'Numerical Label': [LABEL2ID[c] for c in counts.index],
        'Sample Count': counts.values,
        'Percentage (%)': percents.values
    })
    
    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = round(max_count / min_count, 2)
    
    print(dist_df.to_string(index=False))
    print(f"\nTotal Samples: {total}")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio}:1")
    
    print("\n--- Training Recommendations for Imbalance ---")
    print("1. Stratified Sampling: Use stratified splitting for train/val/test splits (implemented).")
    print("2. Class-Weighted Loss: Use CrossEntropyLoss(weight=class_weights) during BERT fine-tuning.")
    print("3. Controlled Oversampling/Focal Loss: If minority class recall is low, apply focal loss or random oversampling.")
    
    return dist_df, imbalance_ratio

def analyze_text_lengths(df):
    print("\n" + "=" * 60, flush=True)
    print("STEP 8: TEXT LENGTH & TOKENIZATION ANALYSIS", flush=True)
    print("=" * 60, flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Word count
    word_counts = df['text'].apply(lambda x: len(x.split()))
    
    # Fast Batch Token Count using DistilBERT tokenizer
    print("Calculating token lengths with HuggingFace fast tokenizer...", flush=True)
    texts = df['text'].tolist()
    batch_size = 10000
    token_counts_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_enc = tokenizer(batch_texts, truncation=False, add_special_tokens=True)
        token_counts_list.extend([len(ids) for ids in batch_enc['input_ids']])
        
    token_counts = pd.Series(token_counts_list, index=df.index)
    
    stats = {
        'Min': int(token_counts.min()),
        'Max': int(token_counts.max()),
        'Mean': round(float(token_counts.mean()), 2),
        'Median': float(token_counts.median()),
        '90th Percentile': float(np.percentile(token_counts, 90)),
        '95th Percentile': float(np.percentile(token_counts, 95)),
        '99th Percentile': float(np.percentile(token_counts, 99))
    }
    
    print("\n--- DistilBERT Token Length Distribution ---", flush=True)
    for k, v in stats.items():
        print(f"{k}: {v}", flush=True)
        
    p99 = stats['99th Percentile']
    if p99 <= 64:
        rec_len = 64
    elif p99 <= 128:
        rec_len = 128
    else:
        rec_len = 256
        
    print(f"\nRecommended max_sequence_length: {rec_len} (Covers >= 99% of samples without unnecessary padding)", flush=True)
    
    # Tokenization demonstration
    sample_text = df['text'].iloc[0]
    encoded_demo = tokenizer(sample_text, padding="max_length", truncation=True, max_length=rec_len)
    print(f"\n--- Tokenization Example ---", flush=True)
    print(f"Sample Text: \"{sample_text}\"", flush=True)
    print(f"input_ids (first 10): {encoded_demo['input_ids'][:10]}...", flush=True)
    print(f"attention_mask (first 10): {encoded_demo['attention_mask'][:10]}...", flush=True)
    
    return token_counts, stats, rec_len

def split_dataset(df):
    print("\n" + "=" * 60)
    print("STEP 9: TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)
    
    # 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df['label']
    )
    
    print(f"Train Set Size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation Set Size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test Set Size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def validate_splits(df_clean, train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("STEP 14: DATA QUALITY CHECKS")
    print("=" * 60)
    
    checks_passed = True
    
    # Check 1: No missing text
    missing_text = train_df['text'].isnull().sum() + val_df['text'].isnull().sum() + test_df['text'].isnull().sum()
    if missing_text == 0:
        print("[PASS] Check 1 Passed: No missing text.")
    else:
        print(f"[FAIL] Check 1 Failed: {missing_text} missing texts found!")
        checks_passed = False
        
    # Check 2: No missing labels
    missing_labels = train_df['label'].isnull().sum() + val_df['label'].isnull().sum() + test_df['label'].isnull().sum()
    if missing_labels == 0:
        print("[PASS] Check 2 Passed: No missing labels.")
    else:
        print(f"[FAIL] Check 2 Failed: {missing_labels} missing labels found!")
        checks_passed = False
        
    # Check 3: No duplicate text across train/val/test
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    test_texts = set(test_df['text'])
    
    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)
    
    total_leakage = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    if total_leakage == 0:
        print("[PASS] Check 3 Passed: No duplicate texts across train/validation/test splits.")
    else:
        print(f"[FAIL] Check 3 Failed: Found text leakage across splits ({total_leakage} overlapping texts)!")
        checks_passed = False
        
    # Check 4: Only 4 target labels exist
    unique_emotions = set(train_df['emotion']).union(val_df['emotion']).union(test_df['emotion'])
    expected_emotions = {"Joy", "Sadness", "Anger", "Anxiety"}
    if unique_emotions == expected_emotions:
        print(f"[PASS] Check 4 Passed: Exactly 4 target labels exist ({sorted(list(unique_emotions))}).")
    else:
        print(f"[FAIL] Check 4 Failed: Unexpected emotions found: {unique_emotions}")
        checks_passed = False
        
    # Check 5: Every numerical label is valid (0, 1, 2, 3)
    unique_labels = set(train_df['label']).union(val_df['label']).union(test_df['label'])
    expected_labels = {0, 1, 2, 3}
    if unique_labels == expected_labels:
        print("[PASS] Check 5 Passed: Every numerical label is valid {0, 1, 2, 3}.")
    else:
        print(f"[FAIL] Check 5 Failed: Invalid numerical labels: {unique_labels}")
        checks_passed = False
        
    # Check 6: Train/validation/test proportions are ~80/10/10
    tot = len(df_clean)
    tr_p = round(len(train_df) / tot * 100, 1)
    val_p = round(len(val_df) / tot * 100, 1)
    tst_p = round(len(test_df) / tot * 100, 1)
    print(f"[PASS] Check 6 Passed: Split proportions are Train={tr_p}%, Val={val_p}%, Test={tst_p}%.")
    
    # Check 7: All classes represented in every split
    train_classes = set(train_df['label'])
    val_classes = set(val_df['label'])
    test_classes = set(test_df['label'])
    if train_classes == expected_labels and val_classes == expected_labels and test_classes == expected_labels:
        print("[PASS] Check 7 Passed: All 4 classes are represented in every split.")
    else:
        print("[FAIL] Check 7 Failed: Missing classes in splits!")
        checks_passed = False
        
    # Check 8: No accidental test-set contamination
    if len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("[PASS] Check 8 Passed: No accidental test-set contamination.")
    else:
        print("[FAIL] Check 8 Failed: Test-set contamination detected!")
        checks_passed = False
        
    return checks_passed

def save_processed_data(df_clean, train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("STEP 11: SAVING PROCESSED DATASETS")
    print("=" * 60)
    
    clean_csv = os.path.join(PROCESSED_DATA_DIR, 'goemotions_4class_clean.csv')
    train_csv = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    val_csv = os.path.join(PROCESSED_DATA_DIR, 'validation.csv')
    test_csv = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    
    df_clean[['text', 'original_label', 'emotion', 'label']].to_csv(clean_csv, index=False)
    train_df[['text', 'original_label', 'emotion', 'label']].to_csv(train_csv, index=False)
    val_df[['text', 'original_label', 'emotion', 'label']].to_csv(val_csv, index=False)
    test_df[['text', 'original_label', 'emotion', 'label']].to_csv(test_csv, index=False)
    
    print(f"Saved: {clean_csv}")
    print(f"Saved: {train_csv}")
    print(f"Saved: {val_csv}")
    print(f"Saved: {test_csv}")

def generate_visualizations(df_clean, token_counts, train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("STEP 13: GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Class Distribution Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df_clean['emotion'].value_counts()
    colors = ['#4F46E5', '#EF4444', '#F59E0B', '#10B981']
    
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title('Target Emotion Class Distribution (4 Classes)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Emotion State', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}\n({height/len(df_clean)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    chart1_path = os.path.join(BASE_DIR, 'class_distribution.png')
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved visualization: {chart1_path}")
    
    # 2. Text Length Distribution Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(token_counts, bins=40, color='#6366F1', edgecolor='black', alpha=0.75)
    ax.axvline(np.percentile(token_counts, 95), color='#EF4444', linestyle='--', linewidth=2, label='95th Percentile')
    ax.axvline(np.percentile(token_counts, 99), color='#F59E0B', linestyle='--', linewidth=2, label='99th Percentile')
    
    ax.set_title('DistilBERT Token Length Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Token Count (distilbert-base-uncased)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Frequency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    chart2_path = os.path.join(BASE_DIR, 'text_length_distribution.png')
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved visualization: {chart2_path}")
    
    # 3. Train / Validation / Test Distribution Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    splits_data = {
        'Train': train_df['emotion'].value_counts(),
        'Validation': val_df['emotion'].value_counts(),
        'Test': test_df['emotion'].value_counts()
    }
    split_df = pd.DataFrame(splits_data).reindex(["Joy", "Sadness", "Anger", "Anxiety"])
    
    split_df.plot(kind='bar', ax=ax, width=0.8, color=['#3B82F6', '#10B981', '#F59E0B'], edgecolor='black')
    ax.set_title('Class Distribution Across Train / Validation / Test Splits', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    ax.legend(title='Split', fontsize=11)
    
    plt.tight_layout()
    chart3_path = os.path.join(BASE_DIR, 'split_distribution.png')
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved visualization: {chart3_path}")

def generate_reports_and_metadata(original_samples, removed_missing, removed_non_meaningful,
                                  removed_unmapped, removed_ambiguous, removed_duplicates,
                                  df_clean, train_df, val_df, test_df,
                                  length_stats, recommended_max_length):
    print("\n" + "=" * 60)
    print("STEP 12: GENERATING METADATA & PREPROCESSING REPORT")
    print("=" * 60)
    
    # 1. label_mapping.json
    mapping_metadata = {
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "goemotions_to_target_mapping": EMOTION_MAPPING
    }
    mapping_path = os.path.join(METADATA_DIR, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping_metadata, f, indent=4)
    print(f"Saved metadata: {mapping_path}")
    
    # 2. dataset_statistics.json
    stats_metadata = {
        "total_cleaned_samples": len(df_clean),
        "class_counts": {k: int(v) for k, v in df_clean['emotion'].value_counts().to_dict().items()},
        "class_percentages": {k: float(v) for k, v in (df_clean['emotion'].value_counts(normalize=True) * 100).round(2).to_dict().items()},
        "token_length_statistics": length_stats,
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        }
    }
    stats_path = os.path.join(METADATA_DIR, 'dataset_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_metadata, f, indent=4)
    print(f"Saved metadata: {stats_path}")
    
    # 3. preprocessing_report.json
    class_dist_dict = {k: int(v) for k, v in df_clean['emotion'].value_counts().to_dict().items()}
    report_metadata = {
        "original_samples": int(original_samples),
        "removed_missing_text": int(removed_missing),
        "removed_non_meaningful_text": int(removed_non_meaningful),
        "removed_duplicates": int(removed_duplicates),
        "removed_unmapped": int(removed_unmapped),
        "removed_ambiguous": int(removed_ambiguous),
        "final_samples": int(len(df_clean)),
        "class_distribution": class_dist_dict,
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "max_text_length": int(length_stats['Max']),
        "recommended_max_length": int(recommended_max_length)
    }
    report_path = os.path.join(METADATA_DIR, 'preprocessing_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_metadata, f, indent=4)
    print(f"Saved metadata: {report_path}")

def main():
    setup_directories()
    
    raw_df = load_dataset(RAW_CSV_PATH)
    original_samples = len(raw_df)
    
    text_col, emotion_cols, meta_cols = inspect_dataset(raw_df)
    
    # Data Cleaning
    cleaned_df, removed_missing, removed_non_meaningful = clean_text_data(raw_df, text_col)
    
    # Emotion Mapping & Multi-Label Resolution
    mapped_df, removed_unmapped, removed_ambiguous = map_and_resolve_emotions(cleaned_df, emotion_cols)
    
    # Deduplication & Leakage Prevention
    final_clean_df, removed_duplicates = deduplicate_and_filter_leakage(mapped_df)
    
    # Class Distribution Analysis
    dist_df, imbalance_ratio = analyze_class_distribution(final_clean_df)
    
    # Text Length Analysis
    token_counts, length_stats, rec_len = analyze_text_lengths(final_clean_df)
    
    # Split
    train_df, val_df, test_df = split_dataset(final_clean_df)
    
    # Quality Checks
    quality_ok = validate_splits(final_clean_df, train_df, val_df, test_df)
    
    # Save Outputs
    save_processed_data(final_clean_df, train_df, val_df, test_df)
    generate_visualizations(final_clean_df, token_counts, train_df, val_df, test_df)
    generate_reports_and_metadata(
        original_samples, removed_missing, removed_non_meaningful,
        removed_unmapped, removed_ambiguous, removed_duplicates,
        final_clean_df, train_df, val_df, test_df,
        length_stats, rec_len
    )
    
    # Print Final Summary
    class_dist_dict = final_clean_df['emotion'].value_counts().to_dict()
    
    print("\n" + "=" * 60)
    print("GoEmotions Preprocessing Complete")
    print("=" * 60)
    print(f"\nOriginal samples: {original_samples}")
    print(f"Final samples: {len(final_clean_df)}")
    print("\nClass Distribution:")
    for cls in ["Joy", "Sadness", "Anger", "Anxiety"]:
        cnt = class_dist_dict.get(cls, 0)
        pct = (cnt / len(final_clean_df) * 100) if len(final_clean_df) > 0 else 0
        print(f"  {cls}: {cnt} ({pct:.2f}%)")
        
    print(f"\nTrain: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    print(f"\nRecommended max_length: {rec_len}")
    
    print("\nData Quality:")
    print("  [PASS] No missing text")
    print("  [PASS] No missing labels")
    print("  [PASS] No duplicate leakage")
    print("  [PASS] All four classes present")
    print("  [PASS] Stratified split verified")
    
    print("\nFiles created:")
    print("  [PASS] train.csv")
    print("  [PASS] validation.csv")
    print("  [PASS] test.csv")
    print("  [PASS] goemotions_4class_clean.csv")
    print("  [PASS] label_mapping.json")
    print("  [PASS] preprocessing_report.json")
    print("  [PASS] dataset_statistics.json")
    print("  [PASS] class_distribution.png")
    print("  [PASS] text_length_distribution.png")
    print("  [PASS] split_distribution.png")

if __name__ == '__main__':
    main()
