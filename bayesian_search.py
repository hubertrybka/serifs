import argparse
import multiprocessing as mp
import os
import queue
import random
import time
import warnings
import configparser

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

from serifs.clf.scorer import SKLearnScorer


# suppress scikit-learn warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn


def search(config_path, return_list):
    """
    Perform Bayesian optimization on the latent space in respect to the discriminator output
    Args:
        config_path: path to the config file
        return_list: list to append results to (multiprocessing)
    Returns:
        None
    """

    # read config file

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)
    model_path = config['SEARCH']['model_path']
    latent_size = int(config['SEARCH']['latent_size'])
    n_init = int(config['SEARCH']['n_init'])
    n_iter = int(config['SEARCH']['n_iter'])
    bounds = float(config['SEARCH']['bounds'])
    verbosity = int(config['SEARCH']['verbosity'])

    # initialize scorer
    latent_size = latent_size
    scorer = SKLearnScorer(model_path, penalize=False)

    # define bounds
    pbounds = {str(p): (-bounds, bounds) for p in range(latent_size)}

    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.2)

    # initialize optimizer
    optimizer = BayesianOptimization(
        f=scorer,
        pbounds=pbounds,
        random_state=(time.time_ns() % 10 ** 6),
        verbose=verbosity > 1,
        bounds_transformer=bounds_transformer
    )

    vector_list = []
    score_list = []

    # run optimization:
    optimizer.maximize(
        init_points=n_init,
        n_iter=n_iter,
    )
    vector = np.array(list(optimizer.max['params'].values()))

    score_list.append(float(optimizer.max['target']))
    vector_list.append(vector)

    # append results to return list

    samples = pd.DataFrame(np.array(vector_list))
    samples.columns = [str(n) for n in range(latent_size)]
    samples['score'] = score_list
    samples['score'] = samples['score'].astype(float)
    samples['norm'] = np.linalg.norm(samples.iloc[:, :-1], axis=1)
    return_list.append(samples)
    return None


if __name__ == '__main__':
    random.seed(42)
    """
    Multiprocessing support and queue handling
    """
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config_files/search_config.ini')
    config_path = parser.parse_args().config

    # read config file
    config = configparser.ConfigParser()
    config.read(config_path)

    n_workers = int(config['SEARCH']['n_workers'])
    verbosity = int(config['SEARCH']['verbosity'])
    n_samples = int(config['SEARCH']['n_samples'])
    n_init = int(config['SEARCH']['n_init'])
    n_iter = int(config['SEARCH']['n_iter'])
    bounds = float(config['SEARCH']['bounds'])
    latent_size = int(config['SEARCH']['latent_size'])
    model_path = config['SEARCH']['model_path']

    samples = pd.DataFrame()  # placeholder

    manager = mp.Manager()
    return_list = manager.list()
    cpu_cores = mp.cpu_count()
    if n_workers != -1:
        cpus = n_workers if n_workers < cpu_cores else cpu_cores
    else:
        cpus = cpu_cores

    print(f"Bayesian search started succesfully") if verbosity > 0 else None
    print("Number of workers: ", cpus) if verbosity > 0 else None

    queue = queue.Queue()

    for i in range(n_samples):
        proc = mp.Process(target=search, args=[config_path, return_list])
        queue.put(proc)

    print('(mp) Processes in queue: ', queue.qsize()) if verbosity > 0 else None

    queue_initial_size = queue.qsize()
    if queue_initial_size >= 1000:
        period = 100
    if queue_initial_size >= 500:
        period = 50
    elif queue_initial_size >= 100:
        period = 20
    else:
        period = 5

    # handle the queue

    while True:
        processes = []
        if queue.empty():
            print("(mp) Queue handled successfully") if verbosity > 0 else None
            break
        while len(mp.active_children()) < cpus:
            if queue.empty():
                break
            proc = queue.get()
            proc.start()
            if queue.qsize() % period == 0:
                print('(mp) Processes in queue: ', queue.qsize()) if verbosity > 0 else None
            processes.append(proc)

            # complete the processes
        for proc in processes:
            proc.join()
        time.sleep(1)

    samples = pd.concat(return_list)
    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60  # in minutes
    if time_elapsed < 60:
        print("Time elapsed: ", round(time_elapsed, 2), "min") if verbosity > 0 else None
    else:
        print("Time elapsed: ",
              int(time_elapsed // 60), "h",
              round(time_elapsed % 60, 2), "min") if verbosity > 0 else None

    # save the results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = 'bayesian_search_' + timestamp

    # create results directory
    os.mkdir(f'outputs/{model_name}')

    # save the results
    samples.to_csv(f'outputs/{model_name}/latent_vectors.csv', index=False)

    print(f"Results saved to: outputs/{model_name}") if verbosity > 0 else None

    # save the arguments
    with open(f'outputs/{model_name}/info.txt', 'w') as f:
        text = [f'model_path: {model_path}',
                f'latent_size: {latent_size}',
                f'n_samples: {n_samples}',
                f'init_points: {n_init}',
                f'n_iter: {n_iter}',
                f'bounds: {bounds}',
                f'verbosity: {verbosity}',
                f'time elapsed per sample: {round(time_elapsed / n_samples, 2)} min',
                f'mean score: {round(samples["score"].mean(), 2)}',
                f'sigma score: {round(samples["score"].std(), 2)}',
                f'mean norm: {round(samples["norm"].mean(), 2)}',
                f'sigma norm: {round(samples["norm"].std(), 2)}']
        text = '\n'.join(text)
        f.write(text)
