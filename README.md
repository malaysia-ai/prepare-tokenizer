# prepare-tokenizer

Prepare SentencePiece (T5, Llama2) and Byte level (GPT2, RoBERTa) BPE on Malaysian texts (Jawi, Melayu, Manglish, Mandarin, Indian).

## how-to train

1. Combine https://huggingface.co/datasets/malaysia-ai/dedup-text-dataset into 1 text file, [combine.ipynb](combine.ipynb).

```bash
ls -lh combine.txt
```

```
-rwxrwxrwx 1 ubuntu ubuntu 53G Oct 17 15:48 combine.txt
```

2. Train SentencePiece, [train-sentencepiece.ipynb](train-sentencepiece.ipynb).

SentencePiece use for T5 and Llama2.

**We use Standard_HB60-15rs to train**.

3. Train BPE, [train-bpe.ipynb](train-bpe.ipynb).

BPE use for GPT2 and RoBERTa.

**We use Standard_HB60-15rs to train**.