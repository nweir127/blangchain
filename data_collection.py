import asyncio
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import csv 
import pandas as pd
from langchain import PromptTemplate
from tqdm import tqdm
from transformers import HfArgumentParser
from src.generators.openai_gpt import SimplePromptOpenAIGenerator, JSONItemGenerator
from src.utils import flatten
import argparse
import json
import re
# from datasets import load_dataset


os.chdir("..")


PROMPT = """For each of the following questions, we have asked a reasoning system to give an answer and justify its decision in the form of "BECAUSE ____ AND ____ IT FOLLOWS THAT _____.

Your job is to help debug the model's line of reasoning. Are the individual facts that it used correct? 
For each fact in the answer, provide a score between 1 and 5 reflecting how often the statement holds true in the wild.
1 is "never true"
2 is "rarely true"
3 is "sometimes true"
4 is "usually true"
5 is "always true"

Your output format is a serialized json on a single line with the format {{"<fact1>": <fact1_score>, "fact2": <fact2_score>}} and nothing else.  


QUESTION 1:
Which is an example of conduction? (A) a space heater turned on (B) water boiling on the stove (C) sunlight shining through the window (D, CORRECT) a metal spoon warming in a pot of hot soup

SYSTEM RESPONSE 1:
BECAUSE a space heater is a kind of electrical device
AND an electrical device turns on / off
IT FOLLOWS THAT a space heater turned on is an example of conduction

DEBUG 1:
{{"a space heater is a kind of electrical device": 5, "an electrical device turns on / off": 4}}

QUESTION 2:
How do the spines of a cactus help it survive? (A) Spines help the cactus get moisture. (B) Spines anchor the cactus in the ground. (C, CORRECT) Spines protect the cactus from animals. (D) Spines support the stems and branches of the cactus.

RESPONSE 2:
BECAUSE the spines of a cactus help it survive by keeping predators away
AND keeping predators away helps a cactus get moisture
IT FOLLOWS THAT the spines of a cactus help it survive by the cactus get moisture

DEBUG 2:
{{"the spines of a cactus help it survive by keeping predators away": 5, "keeping predators away helps a cactus get moisture": 2}}

QUESTION 3:
{question}

RESPONSE 3:
{response}

DEBUG 3:
"""


class FactScoreGenerator(SimplePromptOpenAIGenerator, JSONItemGenerator):
    def __init__(self, **kwargs):
        prompt_template = PromptTemplate.from_template(PROMPT)
        super(FactScoreGenerator, self).__init__(prompt_template=prompt_template, **kwargs)
    
    def get_score_by_key(self, scores, i, k):
        try:
            if k in scores[i][0]:
                return scores[i][0][k]
            elif k[:-1] in scores[i][0]:
                return scores[i][0][k[:-1]]
            else:
                return scores[i][0][k+'.']
        except Exception as e:
            print(e)
            return 10
        
        
    def get_score(self, scores, k, n_outputs):
        return min(self.get_score_by_key(scores, i, k) for i in range(n_outputs))

    async def score_facts(self, question_text: str, premises: List[str], hypothesis: str, n_outputs=3):
        inputs = [dict(
            question=question_text,
            response="BECAUSE {} ".format(premises[0]) + \
                     ' '.join('AND {}'.format(p) for p in premises[1:]) + \
                     " IT FOLLOWS THAT {}".format(hypothesis)
        )]

        generation = (await self.agenerate(inputs, n=n_outputs))[0]
        scores = [self.postprocess_generation(g) for g in generation]
        print("-------------------------------------------------------------------")
        print(scores)

        keys = None
        for score in scores:
            if score:
                keys = score[0]
                break
        if keys:
            min_scores = {k: self.get_score(scores, k, n_outputs) for k in keys}
            return min_scores
        else:
            min_scores = {}


if __name__ == "__main__":
    @dataclass
    class CompositionalEntailmentArguments:
        model: str = field(default="chatgpt")
        out_dir: str = field(default='tmp')
        max_batches: int = field(default=10000)
        dataset: str = field(default='entailment')

    (args,) = HfArgumentParser(CompositionalEntailmentArguments).parse_args_into_dataclasses()

    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )

    ### replace with dataset iterator
    ### because it's async, you can run this in batches

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./blangchain/example_scripts/eb-train-right-answer-entailment.jsonl")
    parser.add_argument("--ak_dataset", default="./guided_inference/data/ARC/ARC-ALL+declaratives-QA2D.csv")
    parser.parse_args()
    cmd_args = parser.parse_args()

    with open(cmd_args.ak_dataset, newline='') as f:
        reader = csv.DictReader(f)
        answer_key = {row.pop('questionID'): row for row in reader}

    with open(cmd_args.dataset , 'r') as json_file:
        json_list = list(json_file)

    dataset = []

    for json_str in json_list:
        result = json.loads(json_str)
        dict_id = result['id']
        result['question'] = answer_key[dict_id.split("-")[0]]['question']
        ans_option = "(" + answer_key[dict_id.split("-")[0]]['AnswerKey'] + ")"
        ans_option_marked = answer_key[dict_id.split("-")[0]]['AnswerKey'] + ", CORRECT"
        result['question'] = re.sub(ans_option, ans_option_marked, result['question'])
        result['premises'] = [e['text'] for e in result['premises']]
        dataset.append(result)


    scorer = FactScoreGenerator(model=args.model)
    all_outputs = []
    batch_size = 5
    for batch_idx, i in enumerate(tqdm(range(0, len(dataset), batch_size))):
        jobs = [
            scorer.score_facts(premises=inp['premises'], hypothesis=inp['hypothesis'], question_text=inp['question'])
            for inp in dataset[i:i + batch_size]]

        async def _run():
            all_answers = await asyncio.gather(*jobs)
            return all_answers


        all_outputs.extend(asyncio.run(_run()))
        if batch_idx == args.max_batches: break

    # breakpoint()

    Path(f"{args.out_dir}/{args.dataset}").mkdir(parents=True, exist_ok=True)
    # save outputs in a manner that makes sense for you
    # print(all_outputs)
    myFile = open(f'{args.out_dir}/{args.dataset}/ai2_chatgpt_anno_1_1.csv', 'w')
    writer = csv.writer(myFile)
    writer.writerow(['Statement', 'Score'])
    for dictionary in all_outputs:
        for k,v in dictionary.items():
            writer.writerow([k,v])
    myFile.close()
