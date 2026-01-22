import pandas as pd
import os
import numpy as np

data = pd.read_csv('../combine.csv', encoding='latin1')
results_h11 = []
results_non_h11 = []

for _, row in data.iterrows():
    case = row['case']
    label = row['label']
    filepath = f'xxxx/{case}.csv'

    if not os.path.exists(filepath):
        print(f"Missing file: {filepath}")
        # H11 results
        results_h11.append({
            'case': case,
            'question_number': 'H11',
            'text': np.nan,
            'label': label
        })
        # non-H11 results, empty since file is missing
        results_non_h11.append({
            'case': case,
            'text': np.nan,
            'label': label
        })
        continue

    df = pd.read_csv(filepath)

    qn_candidates = ['question number', 'questionnumber', 'Question Number', 'QuestionNumber']
    content_candidates = ['content', 'Content']
    speaker_candidates = ['speaker(interviewer/interviewee)', 'Speaker(interviewer/interviewee)', 'speaker']

    def find_column(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    qn_col = find_column(qn_candidates)
    content_col = find_column(content_candidates)
    speaker_col = find_column(speaker_candidates)

    if None in [qn_col, content_col, speaker_col]:
        print(f"Missing fields in file: {filepath}")
        results_h11.append({
            'case': case,
            'question_number': 'H11',
            'text': np.nan,
            'label': label
        })
        results_non_h11.append({
            'case': case,
            'text': np.nan,
            'label': label
        })
        continue

    # Find indices where question numbers are not NaN
    qn_indices = df[df[qn_col].notna()].index.tolist()

    # Find the index of H11 question
    h11_idx = None
    for idx in qn_indices:
        if str(df.loc[idx, qn_col]).strip() == 'H11':
            h11_idx = idx
            break

    # Process H11 part
    if h11_idx is not None:
        next_indices = [i for i in qn_indices if i > h11_idx]
        end_idx = next_indices[0] if next_indices else len(df)
        chunk_h11 = df.iloc[h11_idx:end_idx]

        full_text_h11 = ' '.join(chunk_h11[content_col].astype(str))
        interviewee_text_h11 = ' '.join(
            chunk_h11[chunk_h11[speaker_col].str.lower() == 'interviewee'][content_col].astype(str)
        )
    else:
        full_text_h11 = np.nan
        interviewee_text_h11 = np.nan

    results_h11.append({
        'case': case,
        'question_number': 'H11',
        'text': interviewee_text_h11,
        'label': label
    })

    # Process non-H11 part by removing the H11 section
    if h11_idx is not None:
        df_non_h11 = pd.concat([df.iloc[:h11_idx], df.iloc[end_idx:]])
    else:
        df_non_h11 = df.copy()

    # Collect unique question numbers excluding empty
    qn_values = df_non_h11[qn_col].fillna('')
    unique_qns = qn_values[qn_values != ''].unique()

    for qn in unique_qns:
        chunk = df_non_h11[df_non_h11[qn_col] == qn]
        if chunk.empty:
            continue
        full_text_non_h11 = ' '.join(chunk[content_col].astype(str))
        interviewee_text_non_h11 = ' '.join(
            chunk[chunk[speaker_col].str.lower() == 'interviewee'][content_col].astype(str)
        )
        results_non_h11.append({
            'case': case,
            'text': interviewee_text_non_h11,
            'label': label
        })

# Save the two result DataFrames
final_h11_df = pd.DataFrame(results_h11)
final_h11_df.to_csv('H11_data.csv', index=False)

final_non_h11_df = pd.DataFrame(results_non_h11)
final_non_h11_df.to_csv('nonH11_data.csv', index=False)