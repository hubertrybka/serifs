import argparse
import multiprocessing as mp
import os
import queue
import random
import time
import warnings

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

from serifs.clf.scorer import SKLearnScorer


# suppress scikit-learn warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn


def search(args, return_list):
    """
    Perform Bayesian optimization on the latent space in respect to the discriminator output
    Args:
        args: dictionary of arguments (argparse)
            contains:
                model_path: path to the model
                n_samples: number of samples to generate
                init_points: number of initial points to sample
                n_iter: number of iterations to perform
                bounds: bounds for the latent space search
                verbosity: verbosity level
                latent_size: size of the latent space
        return_list: list to append results to (multiprocessing)
    Returns:
        None
    """

    # initialize scorer
    latent_size = args.latent_size
    scorer = SKLearnScorer(args.model_path, penalize=False)

    # define bounds
    pbounds = {str(p): (-args.bounds, args.bounds) for p in range(latent_size)}

    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.2)

    # initialize optimizer
    optimizer = BayesianOptimization(
        f=scorer,
        pbounds=pbounds,
        random_state=(time.time_ns() % 10 ** 6),
        verbose=args.verbosity > 1,
        bounds_transformer=bounds_transformer
    )

    vector_list = []
    score_list = []

    # run optimization:
    optimizer.maximize(
        init_points=args.init_points,
        n_iter=args.n_iter,
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
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to the saved activity predictor model')
    parser.add_argument('-n', '--n_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('-p', '--init_points', type=int, default=8,
                        help='Number of initial points to sample')
    parser.add_argument('-i', '--n_iter', type=int, default=20,
                        help='Number of iterations to perform')
    parser.add_argument('-b', '--bounds', type=float, default=4.0,
                        help='Bounds for the latent space search')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity: 0 - silent, 1 - normal, 2 - verbose')
    parser.add_argument('-l', '--latent_size', type=int, default=32,
                        help='Size of the latent space vector')
    parser.add_argument('-w', '--n_workers', type=int, default=-1,
                        help='Number of workers to use. Default: -1 (all available CPU cores)')

    samples = pd.DataFrame()  # placeholder
    args = parser.parse_args()

    manager = mp.Manager()
    return_list = manager.list()
    cpu_cores = mp.cpu_count()
    if args.n_workers != -1:
        cpus = args.n_workers if args.n_workers < cpu_cores else cpu_cores
    else:
        cpus = cpu_cores

    print("Number of workers: ", cpus) if args.verbosity > 0 else None

    queue = queue.Queue()

    for i in range(args.n_samples):
        proc = mp.Process(target=search, args=[args, return_list])
        queue.put(proc)

    print('(mp) Processes in queue: ', queue.qsize()) if args.verbosity > 0 else None

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
            print("(mp) Queue handled successfully") if args.verbosity > 0 else None
            break
        while len(mp.active_children()) < cpus:
            if queue.empty():
                break
            proc = queue.get()
            proc.start()
            if queue.qsize() % period == 0:
                print('(mp) Processes in queue: ', queue.qsize()) if args.verbosity > 0 else None
            processes.append(proc)

            # complete the processes
        for proc in processes:
            proc.join()
        time.sleep(1)

    samples = pd.concat(return_list)
    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60  # in minutes
    if time_elapsed < 60:
        print("Time elapsed: ", round(time_elapsed, 2), "min") if args.verbosity > 0 else None
    else:
        print("Time elapsed: ",
              int(time_elapsed // 60), "h",
              round(time_elapsed % 60, 2), "min") if args.verbosity > 0 else None

    # save the results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    receptor = args.model_path.split('/')[-1].split('.')[-2]
    model_name = receptor + '_SVC_' + timestamp

    # create results directory
    os.mkdir(f'results/{model_name}')

    # save the results
    samples.to_csv(f'results/{model_name}/latent_vectors.csv', index=False)

    # save the arguments
    with open(f'results/{model_name}/info.txt', 'w') as f:
        text = [f'model_path: {args.model_path}',
                f'latent_size: {args.latent_size}',
                f'n_samples: {args.n_samples}',
                f'init_points: {args.init_points}',
                f'n_iter: {args.n_iter}',
                f'bounds: {args.bounds}',
                f'verbosity: {args.verbosity}',
                f'time elapsed per sample: {round(time_elapsed / args.n_samples, 2)} min',
                f'mean score: {round(samples["score"].mean(), 2)}',
                f'sigma score: {round(samples["score"].std(), 2)}',
                f'mean norm: {round(samples["norm"].mean(), 2)}',
                f'sigma norm: {round(samples["norm"].std(), 2)}']
        text = '\n'.join(text)
        f.write(text)
