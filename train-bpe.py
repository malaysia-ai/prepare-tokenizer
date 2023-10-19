from tokenizers import ByteLevelBPETokenizer, trainers
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

def a():
    for f in tqdm(code_files):
        i = 0
        with open(f) as fopen:
            for l in fopen:
                l = json.loads(l)
                if len(l['r']) < 100:
                    continue

                yield l['r']
                i += 1
                if i >= 2e4:
                    break
    for f in files:
        if f in skip:
            continue
        i = 0
        with open(f) as fopen:
            for l in tqdm(fopen):
                try:
                    l = json.loads(l).strip()
                    yield l
                    i += 1
                    if i >= 3e5:
                        break
                except:
                    pass

tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
tokenizer.train_from_iterator(a(), vocab_size = 32000, show_progress = True, special_tokens = ['<pad>', '<s>', '</s>', '<unk>'])
transformer_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,bos_token='<s>', eos_token='</s>',unk_token='<unk>',pad_token='<pad>'
)
transformer_tokenizer.save_pretrained('./bpe')
tokenizer_hf = AutoTokenizer.from_pretrained('./bpe')
tokenizer_hf.push_to_hub('bpe-tokenizer', organization='malaysia-ai')