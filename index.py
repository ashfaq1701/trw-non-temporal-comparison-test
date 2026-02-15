import argparse
import numpy as np
import pandas as pd

BASE_PATH='/mnt/lustre/users/inf/ms2420/non-temporal-comparison-datasets'

NUM_WALKS=10_000_000
MAX_WALK_LENGTH=80

TEMPORAL_DATASET_PATH=f"{BASE_PATH}/temporal"
THUNDERRW_DATASET_PATH=f"{BASE_PATH}/static_preprocessed_thunderrw"
FLOWWALKER_DATASET_PATH=f"{BASE_PATH}/static_preprocessed_flowwalker"

TEMPEST_OUTPUT_FILE=f"{BASE_PATH}/tempest_walks.txt"
THUNDERRW_OUTPUT_FILE=f"{BASE_PATH}/thunderrw_walks.txt"
FLOWWALKER_OUTPUT_FILE=f"{BASE_PATH}/flowwalker_walks.txt"


def get_flowwalker_metrics(num_walks):
    pass


def get_thunderrw_metrics(num_walks):
    pass


def get_tempest_metrics(num_walks):
    pass


def main(sampling_method, num_walks):
    if sampling_method == "tempest":
        get_tempest_metrics(num_walks)
    elif sampling_method == "thunderrw":
        get_thunderrw_metrics(num_walks)
    elif sampling_method == "flowwalker":
        get_flowwalker_metrics(num_walks)
    else:
        print("Invalid sampling method")
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Non-Temporal Method Comparison")

    parser.add_argument('--sampling_method', type=str, required=True,
                        help='Random walk sampling method (tempest, thunderrw or flowwalker)')
    parser.add_argument('--num_walks', type=int, default=10_000_000,
                        help='Number of walks to sample')

    args = parser.parse_args()

    main(args.sampling_method, args.num_walks)

