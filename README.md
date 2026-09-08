# English-to-Nepali Translation with Custom Transformer

An end-to-end Sequence-to-Sequence (Seq2Seq) Transformer implemented **completely from scratch in PyTorch** for English-to-Nepali neural machine translation.

---

## Benchmark Results

Evaluated on the full unseen test set using [SacreBLEU](https://github.com/mjpost/sacrebleu):

| Metric | Score | Sentences Evaluated |
| :--- | :--- | :--- |
| **BLEU** | **22.01** | 10,812 sentences |
| **chrF** | **44.62** | 10,812 sentences |

---

## Sample Translations

| English Input | Predicted Nepali (Beam Search) |
| :--- | :--- |
| *"It must be viewed as a success."* | यसलाई सफलताको रुपमा हेर्नुपर्छ। |
| *"This work will be continued in the future."* | यो काम भविष्यमा पनि चलिरहनेछ। |
| *"This is not a new problem on the internet."* | इन्टरनेटमा यो समस्या नयाँ होइन। |
| *"He and I had an agreement."* | उनी र म एउटा सहमतिमा थियौं। |

---

## Architecture & Specifications

The model strictly follows the architecture from the paper *"Attention Is All You Need"* (Vaswani et al.):

* **Model Type:** Custom Encoder-Decoder Transformer
* **Embedding Dimension (`d_model`):** 512
* **Attention Heads:** 8 (`d_k = d_v = 64`)
* **Feed-Forward Dimension (`dff`):** 2048
* **Encoder / Decoder Layers:** 6 blocks each
* **Max Sequence Length:** 64 tokens
* **Tokenization:** [SentencePiece](https://github.com/google/sentencepiece) Unigram (16,000 subword vocabulary each for English and Nepali)
* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$)
* **Learning Rate Schedule:** Warmup (4,000 steps) + inverse square-root decay
* **Decoding Strategy:** Custom Beam Search (Beam Width = 4)

---

## Dataset

Trained on the parallel English-Nepali corpus from HuggingFace:
* **Source:** [`iamTangsang/Nepali-to-English-Translation-Dataset`](https://huggingface.co/datasets/iamTangsang/Nepali-to-English-Translation-Dataset)
* **Preprocessing:** SentencePiece subword tokenization, danda (`।`) normalization, length filtering, and special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).

---

## Project Structure

```text
.
├── src/
│   ├── layers.py              # InputEmbedding, PositionalEncoding, ResidualConnection, FeedForward
│   ├── multi_head_attention.py# Scaled Dot-Product Multi-Head Attention
│   ├── encoder.py             # Encoder & EncoderBlock
│   ├── decoder.py             # Decoder & DecoderBlock
│   ├── transformer.py         # Complete MyTransformer architecture
│   ├── lr_scheduler.py        # Transformer Paper Warmup Scheduler
│   └── data/
│       └── dataset.py         # Dataset loading, preprocessing & Tokenizer
│
├── scripts/
│   ├── train.py               # CLI training script
│   ├── beam_search.py         # Beam Search
│   ├── translate.py           # Custom Sentence Translation
│   └── test_bleu_score.py     # BLEU & chrF evaluation
│
├── notebooks/
│   └── Transformers.ipynb     # Main Notebook
│
├── tokenizers/                # SentencePiece models (.model, .vocab)
└── checkpoints/               # Saved model checkpoints (.pth)
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/<your-username>/english-to-nepali-transformer.git
cd english-to-nepali-transformer
pip install torch sentencepiece sacrebleu datasets pandas tqdm
```

### 2. Translate Custom Text
Translate any English sentence to Nepali from the command line:
```bash
python -m scripts.translate --text "I am going home."
# Output:
# INPUT ENGLISH: I am going home.
# PREDICTED NEPALI: म घर जाँदै छु।
```

### 3. Evaluate Model (BLEU & chrF)
Run evaluation on the full test split:
```bash
python -m scripts.test_bleu_score
```

### 4. Train From Scratch
```bash
python -m scripts.train --epochs 50 --batch_size 16 --d_model 512 --beam_size 4
```
