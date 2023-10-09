from collections import defaultdict
import pandas as pd
import wandb

init_count = (lambda: dict(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
))

cost_fns = {
    'gpt-4': lambda row: row.get('prompt_tokens', 0) / 1000 * .03 + row.get('completion_tokens', 0) / 1000 * .06,
    'gpt-3.5-turbo': lambda row: row.get('prompt_tokens', 0) / 1000 * .0015 + row.get('completion_tokens', 0) / 1000 * .002,
    'gpt-3.5-turbo-0613': lambda row: row.get('prompt_tokens', 0) / 1000 * .0015 + row.get('completion_tokens', 0) / 1000 * .002,
    'gpt-3.5-turbo-0613-16k': lambda row: row.get('prompt_tokens', 0) / 1000 * .003 + row.get('completion_tokens', 0) / 1000 * .004,
    'gpt-3.5-turbo-16k': lambda row: row.get('prompt_tokens', 0) / 1000 * .003 + row.get('completion_tokens', 0) / 1000 * .004,
    'gpt-3.5-turbo-instruct': lambda row: row.get('prompt_tokens', 0) / 1000 * .0015 + row.get('completion_tokens', 0) / 1000 * .002,
    'text-davinci-003': lambda row: row.get('total_tokens', 0) / 1000 * .02
}


class TokensTracker:
    '''
    Tracks the number of tokens used by each model.
    '''
    counter = defaultdict(init_count)
    module_counter = defaultdict(lambda: defaultdict(init_count))
    total_calls = 0
    total_calls_by_module = defaultdict(int)
    total_cost = 0

    @staticmethod
    def update(llm_output, module=None):
        '''
        Update the token usage counter with the given llm_output.
        :param llm_output:
        :param module:
        :return:
        '''
        model = llm_output.get('model_name', 'none')
        usage_data = llm_output.get('token_usage', {})
        for k, v in usage_data.items():
            TokensTracker.counter[model][k] += v
            if module is not None:
                TokensTracker.module_counter[module][model][k] += v
        TokensTracker.total_calls += 1
        TokensTracker.total_calls_by_module[module] += 1
        if usage_data:
            TokensTracker.total_cost += cost_fns[model](usage_data)

    @staticmethod
    def report(_print=print):
        '''
        prettyprint the current usage statistics as a table
        :param _print:
        :return:
        '''
        df = pd.DataFrame(TokensTracker.counter).T.reset_index(names='model')
        df['cost'] = df.apply(lambda row: cost_fns[row['model']](row), axis=1)
        _print(df.to_string())
        # if wandb.run is not None:
        #     wandb.log({'token_usage': df.cost.sum()})

        if len(TokensTracker.module_counter):
            mdfs = []
            for module, counter in TokensTracker.module_counter.items():
                df = pd.DataFrame(counter).T.reset_index(names='model')
                df['module'] = module
                df['cost'] = df.apply(lambda row: cost_fns[row['model']](row), axis=1)
                mdfs.append(df)
            mdf = pd.concat(mdfs)
            _print(mdf.to_string())

        _print(f"Total Calls: {TokensTracker.total_calls}")

    @staticmethod
    def report_to(file):
        with open(file, 'w') as f:
            TokensTracker.report(_print = lambda x: f.write(str(x) + '\n'))

class ErrorsTracker:
    '''
    currently not used
    '''
    error_counter = defaultdict(int)

    @staticmethod
    def update(error_type):
        ErrorsTracker.error_counter[error_type] += 1

    @staticmethod
    def report():
        for k, v in ErrorsTracker.error_counter.items():
            print("{}\t{}".format(k, v))
