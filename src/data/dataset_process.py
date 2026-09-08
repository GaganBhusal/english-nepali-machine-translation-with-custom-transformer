from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_from_disk
import re
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import os

class DatasetProcess:

    def __init__(self, dataset_path=r'./my_opus_books_dataset', max_len = 64, vocab_size=16000):

        # first to download dataset
        # dataset = load_dataset("iamTangsang/Nepali-to-English-Translation-Dataset")
        # dataset.save_to_disk(dataset_path)

        dataset = load_from_disk(dataset_path)

        self.df_train = pd.DataFrame(dataset['train'])
        self.df_val= pd.DataFrame(dataset['validation'])
        self.df_test = pd.DataFrame(dataset['test'])

        self.max_len = max_len
        self.vocab_size = vocab_size

    @staticmethod
    def normalize_danda(text):
        text = text.strip()

        text = re.sub(r'\s+([।?!])', r'\1', text)
        if text and text[-1] not in '।?!.…"\'”)':
            text += '।'


        return text


    @staticmethod
    def save(df):
        nepali = df['source'].tolist()
        english = df['target'].tolist()

        with open("tokenizers/nepali.txt", "w", encoding = "utf-8") as f:
            for line in nepali:
                f.write(line.strip() + "\n")


        with open("tokenizers/english.txt", "w", encoding = "utf-8") as f:
            for line in english:
                f.write(line.strip() + "\n")

    def tokenizer(self, df):
        if not (os.path.exists("tokenizers/english_vocab.model") and os.path.exists("tokenizers/nepali_vocab.model")):


            DatasetProcess.save(df)

            spm.SentencePieceTrainer.train(
                input=["english.txt"], 
                model_prefix="tokenizers/english_vocab",
                vocab_size=self.vocab_size,
                model_type="unigram",
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece="<PAD>",
                unk_piece="<UNK>",
                bos_piece="<BOS>",
                eos_piece="<EOS>",
            )

            spm.SentencePieceTrainer.train(
                input=["nepali.txt"], 
                model_prefix="tokenizers/nepali_vocab",
                vocab_size=self.vocab_size,
                model_type="unigram",
                character_coverage=0.9995,
                byte_fallback=True,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece="<PAD>",
                unk_piece="<UNK>",
                bos_piece="<BOS>",
                eos_piece="<EOS>",
            )

        self.sp_english = spm.SentencePieceProcessor(model_file='tokenizers/english_vocab.model')
        self.sp_nepali = spm.SentencePieceProcessor(model_file='tokenizers/nepali_vocab.model')



    def preprocess_text(self):
        self.df_train["source"] = self.df_train["source"].apply(lambda x: DatasetProcess.normalize_danda(x) if isinstance(x, str) else x)
        self.df_val["source"] = self.df_val["source"].apply(lambda x: DatasetProcess.normalize_danda(x) if isinstance(x, str) else x)
        self.df_test["source"] = self.df_test["source"].apply(lambda x: DatasetProcess.normalize_danda(x) if isinstance(x, str) else x)

        df = pd.concat([self.df_train, self.df_val, self.df_test], ignore_index=True)


        df["len1"] = df['source'].apply(lambda x : len(x.split()))
        df["len2"] = df['target'].apply(lambda x : len(x.split()))

        self.tokenizer(df)


        self.df_train["nepali_encoded"] = self.df_train["source"].apply(lambda x: self.sp_nepali.encode(x, out_type = int) if isinstance(x, str) else x)
        self.df_val["nepali_encoded"] = self.df_val["source"].apply(lambda x: self.sp_nepali.encode(x, out_type = int) if isinstance(x, str) else x)
        self.df_test["nepali_encoded"] = self.df_test["source"].apply(lambda x: self.sp_nepali.encode(x, out_type = int) if isinstance(x, str) else x)

        self.df_train["english_encoded"] = self.df_train["target"].apply(lambda x: self.sp_english.encode(x, out_type = int) if isinstance(x, str) else x)
        self.df_val["english_encoded"] = self.df_val["target"].apply(lambda x: self.sp_english.encode(x, out_type = int) if isinstance(x, str) else x)
        self.df_test["english_encoded"] = self.df_test["target"].apply(lambda x: self.sp_english.encode(x, out_type = int) if isinstance(x, str) else x)

        self.df_train = self.df_train[(self.df_train["english_encoded"].apply(len) <= self.max_len - 1) & (self.df_train["nepali_encoded"].apply(len) <= self.max_len - 1)]
        self.df_val = self.df_val[(self.df_val["english_encoded"].apply(len) <= self.max_len - 1) & (self.df_val["nepali_encoded"].apply(len) <= self.max_len - 1)]
        self.df_test = self.df_test[(self.df_test["english_encoded"].apply(len) <= self.max_len - 1) & (self.df_test["nepali_encoded"].apply(len) <= self.max_len - 1)]


        self.df_train['english_encoded_pad'] = self.df_train['english_encoded'].apply(lambda x: x + [3] + [0] * (self.max_len-len(x)-1))
        self.df_train['nepali_encoded_pad'] = self.df_train['nepali_encoded'].apply(lambda x: [2] + x + [0] * (self.max_len-len(x)-1))
        self.df_train['nepali_target'] = self.df_train['nepali_encoded'].apply(lambda x: x + [3] + [0] * (self.max_len-len(x)-1))

        self.df_test['english_encoded_pad'] = self.df_test['english_encoded'].apply(lambda x: x + [3] + [0] * (self.max_len-len(x)-1))
        self.df_test['nepali_encoded_pad'] = self.df_test['nepali_encoded'].apply(lambda x: [2] + x + [0] * (self.max_len-len(x)-1))
        self.df_test['nepali_target'] = self.df_test['nepali_encoded'].apply(lambda x: x + [3] + [0] * (self.max_len-len(x)-1))

        self.df_val['english_encoded_pad'] = self.df_val['english_encoded'].apply(lambda x: x + [3] + [0] * (self.max_len-len(x)-1))
        self.df_val['nepali_encoded_pad'] = self.df_val['nepali_encoded'].apply(lambda x: [2] + x + [0] * (self.max_len-len(x)-1))
        self.df_val['nepali_target'] = self.df_val['nepali_encoded'].apply(lambda x: x + [3] + [0] * (self.max_len-len(x)-1))   

    def process_dataloader(self, batch_size = 32):
        self.preprocess_text()

        BATCH_SIZE = batch_size

        source_tensor_encoder_train = torch.tensor(self.df_train['english_encoded_pad'].tolist(), dtype=torch.long)
        target_tensor_decoder_train = torch.tensor(self.df_train['nepali_encoded_pad'].tolist(), dtype=torch.long)
        output_tensor_train = torch.tensor(self.df_train['nepali_target'].tolist(), dtype=torch.long)

        source_tensor_encoder_test = torch.tensor(self.df_test['english_encoded_pad'].tolist(), dtype=torch.long)
        target_tensor_decoder_test = torch.tensor(self.df_test['nepali_encoded_pad'].tolist(), dtype=torch.long)
        output_tensor_test = torch.tensor(self.df_test['nepali_target'].tolist(), dtype=torch.long)

        source_tensor_encoder_val = torch.tensor(self.df_val['english_encoded_pad'].tolist(), dtype=torch.long)
        target_tensor_decoder_val = torch.tensor(self.df_val['nepali_encoded_pad'].tolist(), dtype=torch.long)
        output_tensor_val = torch.tensor(self.df_val['nepali_target'].tolist(), dtype=torch.long)

        train_dataset = TensorDataset(source_tensor_encoder_train, target_tensor_decoder_train, output_tensor_train)
        test_dataset = TensorDataset(source_tensor_encoder_test, target_tensor_decoder_test, output_tensor_test)
        val_dataset = TensorDataset(source_tensor_encoder_val, target_tensor_decoder_val, output_tensor_val)

        self.train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        self.val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
        self.test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    
