# prepare-tokenizer

Prepare SentencePiece (T5, Llama2) and Byte level (GPT2, RoBERTa) BPE on Malaysian texts (Jawi, Melayu, Manglish, Mandarin, Tamil).

## how-to

1. SentencePiece,

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('malaysia-ai/sentencepiece-tokenizer')
tokenizer.encode('husein comel')
tokenizer.encode('husein cute')
tokenizer.encode('حسين چوميل')
tokenizer.encode('侯赛因很可爱')
```

2. BPE,

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('malaysia-ai/bpe-tokenizer')
tokenizer.encode('husein comel')
tokenizer.encode('husein cute')
tokenizer.encode('حسين چوميل')
tokenizer.encode('侯赛因很可爱')
tokenizer.encode('ஹுசைன் அழகாக இருக்கிறார்')
```

## how-to train

1. Train SentencePiece,

```bash
python3 train-sentencepiece.py
```

When training SentencePiece,

- Always partitioned long texts.

**We use Standard_HB60-15rs to train**.

2. Train BPE,

```bash
python3 train-bpe.py
```

**We use Standard_HB60-15rs to train**.