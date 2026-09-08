import torch
from src import MyTransformer
from scripts.beam_search import beam_search_decode
import sentencepiece as spm
import argparse

def translate(device, beam_search, text="Hello",max_len=64):

    sp_english = spm.SentencePieceProcessor(model_file="tokenizers/english_vocab.model")
    sp_nepali = spm.SentencePieceProcessor(model_file="tokenizers/nepali_vocab.model")

    model = MyTransformer(
        vocab_size_source=sp_english.vocab_size(),
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

    source_encoded = sp_english.encode(text, out_type=int)
    source_length = len(source_encoded)
    source_encoded += [3] + [0] * (max_len-source_length-1)

    with torch.no_grad():

        source = torch.tensor(source_encoded).unsqueeze(0).to(device)
        predicted_text = beam_search(
                    model,
                    source,
                    device, 
                    sp_nepali.vocab_size(),
                    max_len, 
                    sp_english,
                    sp_nepali
                )
        print(f"INPUT ENGLISH: {text}")
        print(f"PREDICTED NEPALI: {predicted_text}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Translate English to Nepali")
    parser.add_argument("--text", type=str, default="Hello")

    args = parser.parse_args()
    
    translate(
        device,
        beam_search_decode,
        text=args.text
    )