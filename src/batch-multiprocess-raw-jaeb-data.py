#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:14:38 2019

@author: jameno
"""

import os
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import sys
import subprocess as sub
import numpy as np

# %%


def check_and_make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def jaeb_data_subprocessor(
    dataset_location, dataset_names, processed_data_location, file_index,
):

    print("Starting: " + str(file_index))
    dataset_name = dataset_names[file_index]

    # Set the python unbuffered state to 1 to allow stdout buffer access
    # This allows continuous reading of subprocess output
    os.environ["PYTHONUNBUFFERED"] = "1"
    p = sub.Popen(
        [
            "python",
            "src/preprocess_raw_jaeb_data.py",
            "-dataset_location",
            dataset_location,
            "-dataset_name",
            dataset_name,
            "-processed_data_location",
            processed_data_location,
        ],
        stdout=sub.PIPE,
        stderr=sub.PIPE,
    )

    # Continuous write out stdout output
    # for line in iter(p.stdout.readline, b''):
    #    sys.stdout.write(line.decode(sys.stdout.encoding))
    for line in iter(p.stdout.readline, b""):
        sys.stdout.write(line.decode("utf-8"))

    output, errors = p.communicate()
    output = output.decode("utf-8")
    errors = errors.decode("utf-8")

    if errors != "":
        print(errors)

    print("COMPLETED: " + str(file_index))

    return


def read_tsv_file(dataset_sample_location, file_name):
    data_path = dataset_sample_location + file_name
    return pd.read_csv(data_path, sep="\t")


def combine_all_dataset_samples(dataset_sample_location):
    sample_filenames = pd.Series(os.listdir(dataset_sample_location))
    all_data_samples = sample_filenames.apply(lambda x: read_tsv_file(dataset_sample_location, x))
    all_data_samples = pd.concat(all_data_samples.values)

    return all_data_samples


def build_unique_combination_dictionary(all_data_samples):
    # Build Master Dictionary
    # For each unique combination, assign a combination_id and get the sum of all occurrences
    unique_dictionary = all_data_samples.groupby("column_combination")["column_combination_occurrences"].sum()
    unique_dictionary = pd.DataFrame(unique_dictionary).reset_index()
    unique_dictionary["combination_id"] = np.arange(len(unique_dictionary))
    unique_dictionary["description"] = np.nan

    return unique_dictionary


def save_sample_group(x, all_data_samples, column_combination_sample_location):
    combo_id = x["combination_id"]
    combo = x["column_combination"]
    sample_df = all_data_samples[all_data_samples["column_combination"] == combo].dropna(axis=1).reset_index(drop=True)
    sample_filename = "group-{}-column-combination-samples.tsv".format(combo_id)
    sample_df.to_csv(column_combination_sample_location + sample_filename, sep="\t", index=False)

    return


def process_column_combination_results(
    processed_data_location, dataset_sample_location, column_combination_sample_location
):
    print("Building column combination dictionary and sample library...", end="")
    start_time = time.time()

    all_data_samples = combine_all_dataset_samples(dataset_sample_location)
    unique_dictionary = build_unique_combination_dictionary(all_data_samples)
    unique_dictionary.apply(
        lambda x: save_sample_group(x, all_data_samples, column_combination_sample_location), axis=1
    )
    
    all_data_samples.drop_duplicates(subset="column_combination", inplace=True)
    all_data_samples.to_csv(processed_data_location + "sample_all_unique_combinations.tsv", sep="\t", index=False)

    unique_dictionary.to_csv(processed_data_location + "unique_combination_dictionary.tsv", sep="\t", index=False)

    end_time = time.time()
    elapsed_minutes = round((end_time - start_time) / 60, 4)
    elapsed_time_message = "DONE in " + str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)
    return


def main():
    processed_data_location = "data/processed/"
    column_combination_sample_location = processed_data_location + "column-combination-samples/"
    dataset_sample_location = processed_data_location + "dataset-samples/"
    compressed_dataset_location = processed_data_location + "PHI-compressed-data/"
    dataset_location = "data/PHI-adverse-events/"
    dataset_names = os.listdir(dataset_location)

    # Check & make directories
    [
        check_and_make_directory(dir_name)
        for dir_name in [column_combination_sample_location, dataset_sample_location, compressed_dataset_location]
    ]

    # Start Multiprocessing Pool
    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [
        pool.apply_async(
            jaeb_data_subprocessor, args=[dataset_location, dataset_names, processed_data_location, file_index,],
        )
        for file_index in range(len(dataset_names))
    ]

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_minutes = round((end_time - start_time) / 60, 4)
    elapsed_time_message = str(len(pool_array)) + " Pre-processing completed in: " + str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)

    process_column_combination_results(
        processed_data_location, dataset_sample_location, column_combination_sample_location
    )


# %%
if __name__ == "__main__":
    main()
