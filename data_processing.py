import pandas as pd
import os
import numpy as np

# Load the combined metadata CSV
data = pd.read_csv('combine.csv')

# Initialize a list to store processed results
results = []

# Iterate through each case in the metadata
for _, row in data.iterrows():
    case = row['case']
    label = row['label']
    filepath = f'Y:/Rong/transcripts/HAMD/translated_can_HAMD/{case}.csv'

    # Check if the transcript file exists
    if not os.path.exists(filepath):
        print(f"Missing file: {filepath}")
        results.append({
            'case': case,
            'question_number': 'H11',
            'text': np.nan,
            'interviewee_text': np.nan,
            'label': label
        })
        continue

    # Load the transcript CSV
    df = pd.read_csv(filepath)

    # Possible column names for question number, content, and speaker
    qn_candidates = ['question number', 'questionnumber', 'Question Number', 'QuestionNumber']
    content_candidates = ['content', 'Content']
    speaker_candidates = ['speaker(interviewer/interviewee)', 'Speaker(interviewer/interviewee)', 'speaker']

    # Helper function to find the first matching column from candidates
    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    qn_col = find_col(qn_candidates)
    content_col = find_col(content_candidates)
    speaker_col = find_col(speaker_candidates)

    # If any required column is missing, skip this case
    if None in [qn_col, content_col, speaker_col]:
        print(f"Missing column in file: {filepath}")
        results.append({
            'case': case,
            'question_number': 'H11',
            'text': np.nan,
            'interviewee_text': np.nan,
            'label': label
        })
        continue

    # Identify indices where question numbers exist
    qn_indices = df[df[qn_col].notna()].index.tolist()
    h11_idx = None
    for idx in qn_indices:
        if str(df.loc[idx, qn_col]).strip() == 'H11':
            h11_idx = idx
            break

    # Extract H11 section if it exists
    if h11_idx is not None:
        # Find the start of the next question number after H11
        next_indices = [i for i in qn_indices if i > h11_idx]
        end_idx = next_indices[0] if next_indices else len(df)
        chunk = df.iloc[h11_idx:end_idx]

        # Concatenate all text in H11 section
        full_text = ' '.join(chunk[content_col].astype(str))
        # Extract only interviewee's text
        interviewee_text = ' '.join(
            chunk[chunk[speaker_col].str.lower() == 'interviewee'][content_col].astype(str)
        )
    else:
        full_text = np.nan
        interviewee_text = np.nan

    # Append the processed data for this case
    results.append({
        'case': case,
        'question_number': 'H11',
        'text': full_text,
        'interviewee_text': interviewee_text,
        'label': label
    })

# Save the final processed dataset
final_df = pd.DataFrame(results)
final_df.to_csv('H11_data.csv', index=False)
