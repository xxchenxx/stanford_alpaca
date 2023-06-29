import datasets
from transformers import pipeline, set_seed
set_seed(32)
generator = pipeline('text-generation', model="full_training/checkpoint-400", do_sample=True, max_new_tokens=100)

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
results = []
from tqdm import tqdm
for example in tqdm(eval_set):
    # generate here is a placeholder for your models generations
    example["output"] = generator(example["instruction"])
    results.append(example)

import json
json.dump(results, open("half.json", "w"))