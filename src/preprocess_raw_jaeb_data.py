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

# %% Functions


def check_and_make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def check_master_dictionary(column_dictionary_file):
    dictionary_cols = ["#", "combinations", "occurrences", "info"]

    if os.path.exists(column_dictionary_file):
        column_dictionary = pd.read_csv(column_dictionary_file)
        last_entry_index = int(column_dictionary["#"].max())
    else:
        column_dictionary = pd.DataFrame(index=[0], columns=dictionary_cols)
        last_entry_index = int(0)

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
        column_dictionary.loc[last_entry_index, dictionary_cols] = [last_entry_index, combo, combo_occurrences, ""]
        last_entry_index += 1
    else:
        column_dictionary.loc[column_dictionary["combinations"] == combo, "occurrences"] += combo_occurrences

    return column_dictionary, last_entry_index


def save_combination_sample(combo, column_dictionary, flattened_df, column_sample_location):
    sample = flattened_df.loc[flattened_df["column_combination"] == combo].dropna(axis=1).reset_index().sample(1)

    combo_number = str(column_dictionary.loc[column_dictionary["combinations"] == combo, "#"].values[0])
    sample_file_name = column_sample_location + "{}-column-samples.csv".format(combo_number)

    if os.path.exists(sample_file_name):
        sample_list = pd.read_csv(sample_file_name)
        sample_list = sample_list.append(sample)
    else:
        sample_list = sample

    sample_list.to_csv(sample_file_name, index=False)

    return


def save_flattened_dataset_to_parquet(flattened_df, compressed_dataset_location, dataset_name):
    compressed_dataset_name = dataset_name[:-5] + ".parquet"
    flattened_df.to_parquet(compressed_dataset_location + compressed_dataset_name)

    return

def save_flattened_dataset_to_gzip(flattened_df, compressed_dataset_location, dataset_name):
    compressed_dataset_name = dataset_name[:-5] + ".gz"
    flattened_df.to_csv(compressed_dataset_location + compressed_dataset_name, compression='gzip')

    return


def process_dataset(
    column_dictionary_file, column_sample_location, compressed_dataset_location, dataset_location, dataset_name,
):
    column_dictionary, last_entry_index, dictionary_cols = check_master_dictionary(column_dictionary_file)

    dataset_path = dataset_location + dataset_name
    json_data = import_data(dataset_path)

    flattened_data = flatten_data(json_data)
    flattened_data = calculate_column_combinations(flattened_data)

    flattened_df = convert_json_to_df(flattened_data)

    # Get unique key combinations of data
    unique_combinations = flattened_df["column_combination"].unique()

    for combo in unique_combinations:
        column_dictionary, last_entry_index = update_dictionary(
            combo, column_dictionary, dictionary_cols, last_entry_index, flattened_df
        )
        save_combination_sample(combo, column_dictionary, flattened_df, column_sample_location)

    column_dictionary.to_csv(column_dictionary_file, index=False)

    return flattened_df


def main():
    processed_data_location = "data/processed/"
    column_dictionary_file = processed_data_location + "master-column-combination-dictionary.csv"
    column_sample_location = processed_data_location + "column-combination-samples/"
    compressed_dataset_location = processed_data_location + "PHI-compressed-data/"

    # Check & make directories
    [check_and_make_directory(dir_name) for dir_name in [column_sample_location, compressed_dataset_location]]

    dataset_location = "data/PHI-adverse-events/"
    dataset_names = os.listdir(dataset_location)

    for file_index in range(len(dataset_names)):
        print(file_index)
        dataset_name = dataset_names[file_index]

        flattened_df = process_dataset(
            column_dictionary_file, column_sample_location, compressed_dataset_location, dataset_location, dataset_name,
        )

        # save_flattened_dataset_to_parquet(flattened_df, compressed_dataset_location, dataset_name)
        save_flattened_dataset_to_gzip(flattened_df, compressed_dataset_location, dataset_name)

# %%
if __name__ == "__main__":
    main()
