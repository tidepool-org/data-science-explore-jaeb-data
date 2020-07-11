"""
preprocess_raw_jaeb_data.py

Author: Jason Meno

Description:
    Preprocesses Raw Jaeb Loop Study .json files and converts them into
    flattened compressed files

Dependencies:
    - A folder of .json datasets from the Jaeb Loop Study

"""

# %% Imports
import os
import orjson
import flatten_json
import pandas as pd
import argparse

# %% Functions


def get_args():
    codeDescription = "Process a single raw Jaeb Loop .JSON Dataset"

    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument(
        "-column_sample_location",
        dest="column_sample_location",
        default="data/processed/column-combination-samples/",
        help="The location where samples from each column combination are saved",
    )

    parser.add_argument(
        "-column_dictionary_file",
        dest="column_dictionary_file",
        default="data/processed/master-column-combination-dictionary.tsv",
        help="The location of the master dictionary for all columns",
    )

    parser.add_argument(
        "-compressed_dataset_location",
        dest="compressed_dataset_location",
        default="data/processed/PHI-compressed-data/",
        help="The location where compressed processed datasets are saved",
    )

    parser.add_argument(
        "-dataset_location",
        dest="dataset_location",
        default="data/PHI-adverse-events/",
        help="The location where raw datasets are imported from",
    )

    parser.add_argument(
        "-dataset_name", dest="dataset_name", default="", help="The name of the dataset file to process",
    )

    parser.add_argument(
        "--mode",
        dest="_",
        default="_",
        help="Temp PyCharm console space",
    )

    parser.add_argument(
        "--port",
        dest="_",
        default="_",
        help="Temp PyCharm console space",
    )

    args = parser.parse_args()

    return args


def check_master_dictionary(column_dictionary_file):
    dictionary_cols = ["#", "combinations", "occurrences", "info"]

    if os.path.exists(column_dictionary_file):
        column_dictionary = pd.read_csv(column_dictionary_file, sep='\t')
        last_entry_index = int(column_dictionary["#"].max()) + 1
    else:
        column_dictionary = pd.DataFrame(index=[0], columns=dictionary_cols)
        last_entry_index = 0

    return column_dictionary, last_entry_index, dictionary_cols


def import_data(dataset_path):

    with open(dataset_path) as json_reader:
        raw_data = orjson.loads(json_reader.read())

    # Combine all JSON data blobs into a single list
    json_data = []

    for chunk in raw_data:
        json_data += chunk["data"]

    return json_data


def flatten_data(json_data):
    flattened_data = [flatten_json.flatten(json_data[i], separator=".") for i in range(len(json_data))]
    return flattened_data


def calculate_column_combinations(flattened_data):
    # Add set of all keys combined per object
    for flattened_object in flattened_data:
        flattened_object.update({"column_combination": sorted(set(flattened_object))})
    return flattened_data


def convert_json_to_df(flattened_data):
    flattened_df = pd.DataFrame(flattened_data)
    flattened_df = flattened_df.reindex(sorted(flattened_df.columns), axis=1)
    flattened_df["column_combination"] = flattened_df["column_combination"].astype(str)
    return flattened_df


def update_dictionary(combo, column_dictionary, dictionary_cols, last_entry_index, flattened_df):

    combo_occurrences = sum(flattened_df["column_combination"] == combo)

    if combo not in column_dictionary["combinations"].values:
        column_dictionary.loc[last_entry_index, dictionary_cols] = [str(last_entry_index), combo, combo_occurrences, ""]
        last_entry_index += 1
    else:
        column_dictionary.loc[column_dictionary["combinations"] == combo, "occurrences"] += combo_occurrences

    return column_dictionary, last_entry_index


def save_combination_sample(combo, column_dictionary, flattened_df, column_sample_location):
    sample = flattened_df.loc[flattened_df["column_combination"] == combo].dropna(axis=1).reset_index(drop=True).sample(1)

    combo_number = str(column_dictionary.loc[column_dictionary["combinations"] == combo, "#"].values[0])
    sample_file_name = column_sample_location + "{}-column-samples.tsv".format(combo_number)

    if os.path.exists(sample_file_name):
        sample_list = pd.read_csv(sample_file_name, sep='\t')
        sample_list = sample_list.append(sample)
    else:
        sample_list = sample

    sample_list.to_csv(sample_file_name, sep='\t', index=False)

    return


def save_flattened_dataset_to_parquet(flattened_df, compressed_dataset_location, dataset_name):
    compressed_dataset_name = dataset_name[:-5] + ".parquet"
    flattened_df.to_parquet(compressed_dataset_location + compressed_dataset_name)

    return


def save_flattened_dataset_to_gzip(flattened_df, compressed_dataset_location, dataset_name):
    compressed_dataset_name = dataset_name[:-5] + ".gz"
    flattened_df.to_csv(compressed_dataset_location + compressed_dataset_name, sep='\t', compression="gzip")

    return


def process_dataset(processor_args):
    column_dictionary_file = processor_args.column_dictionary_file
    column_sample_location = processor_args.column_sample_location
    compressed_dataset_location = processor_args.compressed_dataset_location
    dataset_location = processor_args.dataset_location
    dataset_name = processor_args.dataset_name

    dataset_path = dataset_location + dataset_name
    json_data = import_data(dataset_path)

    flattened_data = flatten_data(json_data)
    flattened_data = calculate_column_combinations(flattened_data)

    flattened_df = convert_json_to_df(flattened_data)

    # Get unique key combinations of data
    unique_combinations = flattened_df["column_combination"].unique()

    column_dictionary, last_entry_index, dictionary_cols = check_master_dictionary(column_dictionary_file)


    for combo in unique_combinations:
        column_dictionary, last_entry_index = update_dictionary(
            combo, column_dictionary, dictionary_cols, last_entry_index, flattened_df
        )
        save_combination_sample(combo, column_dictionary, flattened_df, column_sample_location)

    column_dictionary.to_csv(column_dictionary_file, sep='\t', index=False)

    # save_flattened_dataset_to_parquet(flattened_df, compressed_dataset_location, dataset_name)
    save_flattened_dataset_to_gzip(flattened_df, compressed_dataset_location, dataset_name)

    return


# %%
if __name__ == "__main__":
    processor_args = get_args()
    process_dataset(processor_args)
