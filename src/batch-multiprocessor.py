#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:14:38 2019

@author: jameno
"""

import os
import time
from multiprocessing import Pool, cpu_count
import sys
import subprocess as sub

#%%


def check_and_make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def jaeb_data_subprocessor(
    column_dictionary_file,
    column_sample_location,
    compressed_dataset_location,
    dataset_location,
    dataset_names,
    file_index,
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
            "-column_sample_location",
            column_sample_location,
            "-column_dictionary_file",
            column_dictionary_file,
            "-compressed_dataset_location",
            compressed_dataset_location,
            "-dataset_location",
            dataset_location,
            "-dataset_name",
            dataset_name,
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


def main():
    processed_data_location = "data/processed/"
    column_dictionary_file = processed_data_location + "master-column-combination-dictionary.tsv"
    column_sample_location = processed_data_location + "column-combination-samples/"
    compressed_dataset_location = processed_data_location + "PHI-compressed-data/"
    dataset_location = "data/PHI-adverse-events/"
    dataset_names = os.listdir(dataset_location)

    # Check & make directories
    [check_and_make_directory(dir_name) for dir_name in [column_sample_location, compressed_dataset_location]]

    # Start Multiprocessing Pool

    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [
        pool.apply_async(
            jaeb_data_subprocessor,
            args=[
                column_dictionary_file,
                column_sample_location,
                compressed_dataset_location,
                dataset_location,
                dataset_names,
                file_index,
            ],
        )
        for file_index in range(len(dataset_names))
    ]

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    elapsed_time_message = str(len(pool_array)) + " Processed completed in: " + str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)


# %%
if __name__ == "__main__":
    main()
