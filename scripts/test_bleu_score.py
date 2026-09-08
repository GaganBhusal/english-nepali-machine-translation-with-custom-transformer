import sacrebleu
import torch
from src.data.dataset_process import DatasetProcess
from src import MyTransformer
from scripts.beam_search import beam_search_decode
import sentencepiece as spm
from datasets import load_from_disk
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def evaluate_translations(device, beam_search, max_len=64, dataset_path=r'./my_opus_books_dataset', batch_size=32):
    hypotheses = []
    references = []

    sp_source = spm.SentencePieceProcessor(model_file="tokenizers/english_vocab.model")
    sp_nepali = spm.SentencePieceProcessor(model_file="tokenizers/nepali_vocab.model")

    dataset = load_from_disk(dataset_path)

    df_test = pd.DataFrame(dataset['test'])

    df_test["source"] = df_test["source"].apply(lambda x: DatasetProcess.normalize_danda(x) if isinstance(x, str) else x)
    df_test["nepali_encoded"] = df_test["source"].apply(lambda x: sp_nepali.encode(x, out_type = int) if isinstance(x, str) else x)
    df_test["english_encoded"] = df_test["target"].apply(lambda x: sp_source.encode(x, out_type = int) if isinstance(x, str) else x)
    df_test = df_test[(df_test["english_encoded"].apply(len) <= max_len - 1) & (df_test["nepali_encoded"].apply(len) <= max_len - 1)]
    df_test['english_encoded_pad'] = df_test['english_encoded'].apply(lambda x: x + [3] + [0] * (max_len-len(x)-1))
    df_test['nepali_encoded_pad'] = df_test['nepali_encoded'].apply(lambda x: [2] + x + [0] * (max_len-len(x)-1))
    df_test['nepali_target'] = df_test['nepali_encoded'].apply(lambda x: x + [3] + [0] * (max_len-len(x)-1))
    source_tensor_encoder_test = torch.tensor(df_test['english_encoded_pad'].tolist(), dtype=torch.long)
    target_tensor_decoder_test = torch.tensor(df_test['nepali_encoded_pad'].tolist(), dtype=torch.long)
    output_tensor_test = torch.tensor(df_test['nepali_target'].tolist(), dtype=torch.long)
    test_dataset = TensorDataset(source_tensor_encoder_test, target_tensor_decoder_test, output_tensor_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)



    model = MyTransformer(
        vocab_size_source=sp_source.vocab_size(),
        vocab_size_target=sp_nepali.vocab_size(),
        seq_length_source=64,
        seq_length_target=64,
        device=device,
        d_model=512,
        heads=8,
        dropout=0.1,
        dff=2048
    ).to(device)

    checkpoint = torch.load("checkpoints/transformer_nepali_checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        for idx, (source, target_input, target_text) in enumerate(test_loader):
            source = source.to(device)

            for i in range(source.size(0)):
                single_source = source[i].unsqueeze(0)
                
                prediction_text = beam_search(
                    model,
                    single_source,
                    device, 
                    sp_nepali.vocab_size(),
                    max_len, 
                    sp_source,
                    sp_nepali
                )

                target_tokens = [tok for tok in target_text[i].tolist() if tok not in [0, 1, 2, 3]]
                target_text_decoded = sp_nepali.decode(target_tokens)

                hypotheses.append(prediction_text)
                references.append(target_text_decoded)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl')
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    print(f"Evaluated on {len(hypotheses)} sentences")
    print(f"BLEU:  {bleu.score:.2f}")
    print(f"chrF:  {chrf.score:.2f}")

    return {'bleu': bleu.score, 'chrf': chrf.score}

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = evaluate_translations(
        device=device,
        beam_search=beam_search_decode,
        max_len=64,
        batch_size = 1
    )