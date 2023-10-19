from tokenizers import SentencePieceBPETokenizer, trainers
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
import orjson as json
import os
from glob import glob
from tqdm import tqdm

files = sorted(glob('dedup-text-dataset/*.jsonl'))
skip = ['NLLB.jsonl', 'common-crawl.jsonl', 'eprints.jsonl']
skip = [os.path.join('dedup-text-dataset', f) for f in skip]

code_files = [
    'code_instructions_120k.jsonl.requested', 
    'python_evol_instruct_51k.jsonl.requested',
    'unnatural-instructions.jsonl.requested'
]

def partition(text, size = 500):
    splitted = text.split()
    return [' '.join(splitted[i: i + size]) for i in range(0, len(splitted), size)]

def a():
    for f in tqdm(code_files):
        i = 0
        with open(f) as fopen:
            for l in fopen:
                l = json.loads(l)
                if len(l['r']) < 100:
                    continue

                partitions = partition(l['r'])
                for p in partitions:
                    yield p

                i += 1
                if i >= 1e4:
                    break
    for f in files:
        if f in skip:
            continue
        i = 0
        with open(f) as fopen:
            for l in tqdm(fopen):
                try:
                    l = json.loads(l).strip()
                    partitions = partition(l)
                    for p in partitions:
                        yield p
                        
                    i += 1
                    if i >= 1e4:
                        break
                except:
                    pass

tokenizer = SentencePieceBPETokenizer()
tokenizer.train_from_iterator(a(), vocab_size = 32000, show_progress = True, special_tokens = ['<pad>', '<s>', '</s>', '<unk>'])
transformer_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,bos_token='<s>', eos_token='</s>',unk_token='<unk>',pad_token='<pad>'
)
transformer_tokenizer.save_pretrained('./sentencepiece')
tokenizer_hf = AutoTokenizer.from_pretrained('./sentencepiece')
tokenizer_hf.push_to_hub('sentencepiece-tokenizer', organization='malaysia-ai')