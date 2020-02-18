import argparse

from tqdm import tqdm

from spectral_metric.config import EMBEDDINGS, make_config
from spectral_metric.visualize import test_job, plot_with_err, visualize


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', type=str, help='List of datasets')
    parser.add_argument('--embd', type=str, choices=EMBEDDINGS, help='Type of embedding to use, if not specified, will use the raw pixels.')
    parser.add_argument('--tsne', action='store_true', help='Whether to use t-sne or not')
    parser.add_argument('--shuffled_class', type=int, default=None, help='Number of class to shuffle')
    parser.add_argument('--small', type=int, default=None, help='Reduce the number of sample per class')
    parser.add_argument('--make_graph', action='store_true', help='Show the dependency graph of the first dataset')
    parser.add_argument('--k_nearest', type=int, default=3, help='k-nn hyperparameter')
    parser.add_argument('--M', type=int, default=100, help='M sample per class')
    return parser.parse_args()


def main():
    args = parse_arg()
    configs = [make_config(k,
                           embd=args.embd,
                           tsne=args.tsne,
                           small=args.small,
                           shuffled_class=args.shuffled_class) for k in args.datasets]

    if args.make_graph:
        visualize(configs[0], k_nearest=args.k_nearest, M_sample=args.M)

    valss = [test_job(config=k, k_nearest=args.k_nearest, M_sample=args.M) for k in tqdm(configs)]
    plot_with_err(*valss)


if __name__ == '__main__':
    main()
