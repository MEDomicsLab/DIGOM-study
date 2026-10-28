"""
Filename: extraction_mimic_notes.py

Author: Hakima Laribi

Description: Defines the classes and functions to extract embeddings from MIMIC-notes
Code mainly borrowed from: https://github.com/lrsoenksen/HAIM/blob/main/MIMIC_IV_HAIM_API.py
"""
import gzip
import os
import shutil
import subprocess
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class MIMICNotesExtractor:
    """Downloads MIMIC-IV-Note from PhysioNet and extracts BERT embeddings of the clinical notes."""

    def __init__(self,
                 path_to_notes: str,
                 to_download: bool = False,
                 base_url: Optional[str] = "https://physionet.org/files/mimic-iv-note/2.2/",
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Args:
            path_to_notes: path to where notes are saved if already downloaded, else where they will be saved
            to_download: boolean to specify if to download the data from physionet
            base_url: URL to physionet repository
            username: username to physionet account (defaults to the PHYSIONET_USERNAME environment variable)
            password: password to physionet account (defaults to the PHYSIONET_PASSWORD environment variable)
        """
        # Set protected attributes
        self._path_to_notes = path_to_notes
        self._to_download = to_download
        self._base_url = base_url
        self._username = username or os.environ.get('PHYSIONET_USERNAME')
        self._password = password or os.environ.get('PHYSIONET_PASSWORD')
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer, self._model = None, None

        # If we need to download the data, get the csv files first and meta-data
        if self._to_download:
            # Make sure download directory exists
            os.makedirs(self._path_to_notes, exist_ok=True)
            # Download all compressed csv files directly in path_to_notes (-nd: no PhysioNet directory tree)
            cmd = ["wget", "-r", "-N", "-c", "-np", "-nd", "-A", "*.csv.gz", "--user", self._username,
                   "--password", self._password, "-P", self._path_to_notes, self._base_url]
            subprocess.run(cmd, check=True)
            # Extract the compressed csv files
            self.extract_compressed_csvs()

    def __call__(self,
                 note_type: Literal['discharge', 'radiology'],
                 models: List[str] = ["emilyalsentzer/Bio_ClinicalBERT"]):
        """
        Extracts one embedding per note with each model and saves them in csv files of up to 100k notes.

        Long notes are split into chunks of at most 510 tokens and the chunk embeddings are averaged.
        Outputs are written in {path_to_notes}/{note_type}/: {model}_features_{note_type}_{k}.csv
        and {model}_features_empty.csv (embedding of an empty string).

        Args:
            note_type: MIMIC-IV-Note table to embed
            models: HuggingFace names of the models to use
        """
        # Create path to where save notes
        os.makedirs(os.path.join(self._path_to_notes, note_type), exist_ok=True)
        # Read corresponding file
        df_notes = pd.read_csv(os.path.join(self._path_to_notes, f'{note_type}.csv'))

        # Load each single model
        for model_name in models:
            model_short_name = model_name.split('/', 1)[-1]
            bert_features = []
            cpt = 0
            print(f"Extracting features with {model_name} ..")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(self._device)

            # Process each single note per patient to have it at a conform length
            processed_notes = []
            for i in tqdm(range(df_notes.shape[0])):
                processed_notes.append(self.split_note_document(df_notes.iloc[i]['text'])[0])

            # Get embeddings of an empty string
            empty_embed, _ = self.get_biobert_embeddings('')
            df_empty_features = pd.DataFrame(empty_embed, columns=[f'feature_{i}' for i in range(len(empty_embed[0]))])
            df_empty_features.to_csv(os.path.join(self._path_to_notes,
                                                  note_type,
                                                  f"{model_short_name}_features_empty.csv"), index=False)

            for idx_note, full_note in tqdm(list(enumerate(processed_notes)), total=len(processed_notes)):
                for i, part_note in enumerate(full_note):
                    # get embeddings per chunk of notes
                    part_embeddings, _ = self.get_biobert_embeddings(part_note)
                    if i == 0:
                        note_embeddings = part_embeddings
                    else:
                        note_embeddings = np.concatenate((note_embeddings, part_embeddings), axis=0)
                # averages all the embeddings obtained for one note
                embedding = note_embeddings.mean(axis=0)
                n_note = df_notes.iloc[idx_note]['note_id']
                n_admission = df_notes.iloc[idx_note]['hadm_id']
                n_patient = df_notes.iloc[idx_note]['subject_id']

                # Append the information
                bert_features.append([n_note, n_admission, n_patient] + embedding.tolist())

                if (idx_note % 100000 == 0) | (idx_note == len(processed_notes) - 1):  # write every 100k iterations
                    print(f"Writing up to {idx_note} note")
                    columns = ['note_id', 'hadm_id', 'subject_id'] + [f'feature_{i}' for i in
                                                                      range(len(bert_features[0]) - 3)]

                    df_bert_features = pd.DataFrame(bert_features, columns=columns)

                    # Save to CSV
                    output_csv = os.path.join(self._path_to_notes,
                                              note_type,
                                              f"{model_short_name}_features_{note_type}_{cpt}.csv")
                    df_bert_features.to_csv(output_csv, index=False)
                    cpt = cpt + 1
                    bert_features = []

    def extract_compressed_csvs(self):
        """
        Extracts all the downloaded compressed csvs
        """
        # Loop through all .csv.gz files
        for filename in os.listdir(self._path_to_notes):
            if filename.endswith(".csv.gz"):
                gz_path = os.path.join(self._path_to_notes, filename)
                csv_path = gz_path[:-3]  # Remove '.gz' extension

                # Extract the .gz file
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(csv_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Delete the original .gz file
                os.remove(gz_path)

    def split_note_document(self, text: str, min_length: int = 15) -> Tuple[List[str], List[int]]:
        """
        Splits a text if too long for embeddings generation (taken from the HAIM repository).
        BERT can only process a string with <= 512 tokens; longer texts are split on line breaks


        Args:
            text: text to be processed into an embedding
            min_length: minimum number of tokens of a text to be kept; shorter texts generally encode
                        minimal information (e.g. 'Name: ___')

        Returns:
            chunk_parse: list of text chunks of at most 510 tokens
            chunk_length: token count of each chunk
        """
        tokens_list = self._tokenizer.tokenize(text)

        if len(tokens_list) <= 510:
            return [text], [1]

        chunk_parse = []
        chunk_length = []
        chunk = text

        # Go through text and aggregate in groups up to 510 tokens (+ padding)
        if len(tokens_list) >= 510:
            temp = chunk.split('\n')
            ind_start = 0
            len_sub = 0
            for i in range(len(temp)):
                temp_tk = self._tokenizer.tokenize(temp[i])
                if len_sub + len(temp_tk) > 510:
                    chunk_parse.append(' '.join(temp[ind_start:i]))
                    chunk_length.append(len_sub)
                    # reset for next chunk
                    ind_start = i
                    len_sub = len(temp_tk)
                else:
                    len_sub += len(temp_tk)
        elif len(tokens_list) >= min_length:
            chunk_parse.append(chunk)
            chunk_length.append(len(tokens_list))

        return chunk_parse, chunk_length

    def get_biobert_embeddings(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtains BERT embeddings of a text (taken from the HAIM repository).

        Args:
            text: input text

        Returns:
            embeddings: pooled embedding of dimension (1, 768)
            hidden_embeddings: last hidden layer of dimension (1, n_tokens, 768)
        """
        tokens_pt = self._tokenizer(text, return_tensors="pt").to(self._device)
        outputs = self._model(**tokens_pt)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_embeddings = last_hidden_state.detach().cpu().numpy()
        embeddings = pooler_output.detach().cpu().numpy()

        return embeddings, hidden_embeddings
