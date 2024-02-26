import asyncio

from tqdm import tqdm


async def batch_gather(jobs, batch_size=50, progbar=False):
    '''
    takes a list of coroutines to await and waits for them in batches (to not overload APIs)

    :param jobs:
    :param batch_size:
    :return:
    '''

    ret = []
    itr = tqdm(range(0, len(jobs), batch_size)) \
        if progbar else range(0, len(jobs), batch_size)
    for i in itr:
        batch = jobs[i:i + batch_size]
        results_i = (await asyncio.gather(*batch))
        ret.extend(results_i)
    return ret
