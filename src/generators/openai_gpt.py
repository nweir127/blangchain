import ast
import copy
import json
import logging
from typing import List, Tuple, Dict

import langchain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import LLMResult
from langchain.schema.language_model import BaseLanguageModel

from src.generators import LMGenerator
from src.generators.async_openai import JitterWaitChatOpenAI
from src.utils.tracking_utils import TokensTracker

logger = logging.getLogger(__name__)
from langchain import OpenAI, LLMChain, PromptTemplate, FewShotPromptTemplate
import asyncio


class OpenAIGenerator(LMGenerator):
    def __init__(self, prompt=None, model='gpt3'):
        """

        :param prompt:
        :param model: either "gpt3" or "Chatgpt"
        """
        self.model_type = model
        self.lm_class: BaseLanguageModel = None
        if model == 'gpt3':
            self.gen_kwargs = {
                "n": 1,
                'temperature': 0.7,
                'model_name': 'text-davinci-003',
                # "top_p": 1,
                "max_tokens": 1000,
                "max_retries": 100,
            }
            self.lm_class = OpenAI

        elif model in ['chatgpt', 'gpt4']:
            self.gen_kwargs = {
                "n": 1,
                'model_name': "gpt-3.5-turbo-0613" if model == 'chatgpt' else 'gpt-4',
                'temperature': 1,
                # "top_p": 1,
                "request_timeout": 600,
                "max_retries": 100,
            }
            # self.lm_class = CachedChatOpenAI
            self.lm_class = JitterWaitChatOpenAI
        else:
            raise NotImplementedError()
        self.batch_size = 50
        self.prompt = prompt
        self.total_tokens = 0

    def generate(self, inputs: List[dict], parallel=False, **gen_kwargs) -> List[List[str]]:
        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type == 'gpt3' and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']
        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)
        ret = []
        for i in range(0, len(inputs), self.batch_size):
            in_batch = inputs[i:i + self.batch_size]
            if parallel:
                async def gen():
                    tasks = [chain.agenerate([ib]) for ib in in_batch]
                    ret_list = await asyncio.gather(*tasks)
                    for lm_out_i in ret_list:
                        logger.info(lm_out_i.llm_output)
                        TokensTracker.update(lm_out_i.llm_output, module=type(self).__name__)
                    return LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list], )

                lm_output = asyncio.run(gen())
            else:
                lm_output = chain.generate(in_batch)
                logger.info(lm_output.llm_output)
                TokensTracker.update(lm_output.llm_output)
            ret.extend([[g.text for g in gen] for gen in lm_output.generations])
        return ret

    async def agenerate(self, inputs: List[dict], **gen_kwargs) -> List[List[str]]:
        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type == 'gpt3' and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']
        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)
        tasks = [chain.agenerate([ib]) for ib in inputs]
        ret_list = await asyncio.gather(*tasks)
        for lm_out_i in ret_list:
            logger.info(lm_out_i.llm_output)
            TokensTracker.update(lm_out_i.llm_output, module=type(self).__name__)
            self.total_tokens += lm_out_i.llm_output.get('token_usage', {}).get('total_tokens', 0)
        lm_output = LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list])

        ret = [[g.text for g in gen] for gen in lm_output.generations]
        return ret

    def format_print(self, input: Dict):
        print(self.prompt.format(**input))


class SimplePromptOpenAIGenerator(OpenAIGenerator):
    def __init__(self, prompt_template: PromptTemplate, model='chatgpt'):
        if model == 'gpt3':
            prompt = prompt_template
        elif model in ['chatgpt', 'gpt4']:
            prompt = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate(prompt=prompt_template)
            ])
        else:
            raise NotImplementedError
        super().__init__(prompt=prompt, model=model)


class JSONItemGenerator:

    async def postprocess_generation(self, gen: str, expected_items: int = None) -> List[dict]:
        """
        Takes a (potentially multi-line) string and turns it into a list of dicts
        """
        results = []

        for line in gen.split('\n'):
            if not line.strip(): continue
            line = line.strip(', ')
            line = line.strip(".")
            try:
                results.append(ast.literal_eval(line.replace('null', "None")))
            except:
                try:
                    results.append(json.loads(line))
                except:
                    try:
                        fixer = JSONFixer()
                        fixed_json: dict = (await fixer.afix(line))
                        results.append(fixed_json)
                    except:
                        continue

        if expected_items and len(results) != expected_items:
            if len(results) > expected_items:
                results = results[:expected_items]
            else:
                res = [{} for _ in range(expected_items)]
                for r in results:
                    res[r['I'] - 1] = r
                if any(res):
                    results = res
                else:  # final resort
                    results = results + [{} for _ in range(expected_items - len(results))]
        return results


class JSONOpenAIGenerator(SimplePromptOpenAIGenerator, JSONItemGenerator):
    def __init__(self, *args, **kwargs):
        super(JSONOpenAIGenerator, self).__init__(*args, **kwargs)

    def batchify(self, items_to_batch, max_size=None):
        if len(items_to_batch) <= 25:
            _statement_batch_size = len(items_to_batch)
        elif len(items_to_batch) > 25 and len(items_to_batch) <= 50:
            _statement_batch_size = int(len(items_to_batch) / 2) + 1
        elif len(items_to_batch) > 50:
            # _statement_batch_size = min(30, int(len(statements_to_score) / 4) + 1)
            _statement_batch_size = 25
        else:
            raise NotImplementedError()
        if max_size is not None:
            if len(items_to_batch) % max_size == 1:
                _statement_batch_size = max_size - 1
            else:
                _statement_batch_size = max_size

        statement_batches = [items_to_batch[i:i + _statement_batch_size]
                             for i in range(0, len(items_to_batch), _statement_batch_size)]

        return statement_batches

    async def run(self, inputs: List[dict], **kwargs) -> List[List[List[dict]]]:
        generations: List[List[str]] = await self.agenerate(inputs, **kwargs)
        result = [list(await asyncio.gather(*[self.postprocess_generation(gg) for gg in g]))
                  for g in generations]
        return result


class JSONFixer(JSONOpenAIGenerator):
    def __init__(self):
        PROMPT = """You are a system for fixing syntax errors in json items. This includes missing quotes around strings and missing closing brackets. If a key is missing its value, map it to None. Do not add new key/value pairs that are not already there.

Given the following malformed json item, return a serialized, one-line version that can be complied by json.loads() in python.
Your output should be this json item on a single line and nothing else. 

{input}
"""
        super(JSONFixer, self).__init__(prompt_template=PromptTemplate.from_template(PROMPT))

    async def afix(self, input_str) -> dict:
        '''
        takes a malformed json line and tries to fix it with gpt
        :param input_str:
        :return: json loaded item
        '''
        inputs = [dict(input=input_str)]
        ret: str = (await self.agenerate(inputs))[0][0]
        ret = ret.strip("\n").split("\n")[0]
        try:
            ret = json.loads(ret)
        except:
            ret = ast.literal_eval(ret.replace('null', "None"))

        if isinstance(ret, str):
            assert False

        return ret


message_type_to_prompt_class = {
    'human': HumanMessagePromptTemplate,
    'ai': AIMessagePromptTemplate
}


class FollowupPromptOpenAIGenerator(OpenAIGenerator):
    def __init__(self, prompt_template_list: List[Tuple[str, PromptTemplate]], model='gpt3'):

        if model == 'gpt3':
            if any(isinstance(i, FewShotPromptTemplate) for i in prompt_template_list[1:]):
                raise NotImplementedError("cannot handle template lists that have fewshot prompts after the first")
            if isinstance(prompt_template_list[0][1], FewShotPromptTemplate):
                combined_template = '\n\n'.join(template.template for (_, template) in prompt_template_list[1:])
                first_prompt: FewShotPromptTemplate = prompt_template_list[0][1]
                prompt = FewShotPromptTemplate(
                    examples=first_prompt.examples,
                    example_selector=first_prompt.example_selector,
                    example_prompt=first_prompt.example_prompt,
                    suffix=first_prompt.suffix + '\n' + combined_template,
                    input_variables=first_prompt.input_variables + PromptTemplate.from_template(
                        combined_template).input_variables,
                    example_separator=first_prompt.example_separator,
                    prefix=first_prompt.prefix
                )
            else:
                def _get_template(t):
                    if isinstance(t, BaseMessagePromptTemplate):
                        return t
                    else:
                        return t.template

                combined_template = '\n\n'.join(template.template for (_, template) in prompt_template_list)
                prompt = PromptTemplate.from_template(combined_template)
        elif model in ['chatgpt', 'gpt4']:
            prompt = ChatPromptTemplate.from_messages([
                message_type_to_prompt_class[_type](prompt=template) for (_type, template) in prompt_template_list
            ])
        else:
            raise NotImplementedError
        super().__init__(prompt=prompt, model=model)
