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
import orjson
import flatten_json
import pandas as pd
import argparse

# %% Functions


def get_args():
    code_description = "Process a single raw Jaeb Loop .JSON Dataset"

    parser = argparse.ArgumentParser(description=code_description)

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
        "-processed_data_location",
        dest="processed_data_location",
        default="data/processed/",
        help="The location where the processed data is saved to",
    )

    parser.add_argument(
        "--mode", dest="_", default="_", help="Temp PyCharm console space",
    )

    parser.add_argument(
        "--port", dest="_", default="_", help="Temp PyCharm console space",
    )

    return parser.parse_args()


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


def calculate_column_combinations(flattened_data, dataset_name):
    all_columns = []
    # Add set of all keys combined per object
    for flattened_object in flattened_data:
        column_combination = sorted(set(flattened_object))
        all_columns += list(column_combination)
        flattened_object.update({"column_combination": column_combination})

    num_unique_columns = len(set(all_columns))
    print(dataset_name + " flattened column size: {}".format(num_unique_columns))
    return flattened_data


def convert_json_to_df(flattened_data):
    flattened_df = pd.DataFrame(flattened_data)
    flattened_df = flattened_df.reindex(sorted(flattened_df.columns), axis=1)
    flattened_df["column_combination"] = flattened_df["column_combination"].astype(str)
    return flattened_df


def get_combination_sample(combo, flattened_df):
    combo_locations = flattened_df["column_combination"] == combo
    sample = flattened_df.loc[combo_locations].sample(1)
    sample["column_combination_occurrences"] = combo_locations.sum()
    return sample


def save_samples(all_samples_df, processed_data_location, dataset_name):
    sample_file_name = dataset_name[:-5] + "-dataset-samples.tsv"
    data_sample_location = processed_data_location + "dataset-samples/"
    all_samples_df.to_csv(data_sample_location + sample_file_name, sep="\t", index=False)


def save_flattened_dataset_to_parquet(flattened_df, processed_data_location, dataset_name):
    compressed_dataset_name = dataset_name[:-5] + ".parquet"
    compressed_save_location = processed_data_location + "PHI-compressed-data/"
    flattened_df.to_parquet(compressed_save_location + compressed_dataset_name)

    return


def save_flattened_dataset_to_gzip(flattened_df, processed_data_location, dataset_name):
    compressed_dataset_name = dataset_name[:-5] + ".gz"
    compressed_save_location = processed_data_location + "PHI-compressed-data/"
    flattened_df.to_csv(compressed_save_location + compressed_dataset_name, sep="\t", compression="gzip")

    return


def process_dataset(dataset_location, dataset_name, processed_data_location):

    dataset_path = dataset_location + dataset_name

    json_data = import_data(dataset_path)
    flattened_data = flatten_data(json_data)
    flattened_data = calculate_column_combinations(flattened_data, dataset_name)
    flattened_df = convert_json_to_df(flattened_data)

    unique_combinations = pd.Series(flattened_df["column_combination"].unique())

    all_samples_df = unique_combinations.apply(lambda x: get_combination_sample(x, flattened_df))
    all_samples_df = pd.concat(all_samples_df.values)
    all_samples_df.reset_index(drop=True, inplace=True)

    save_samples(all_samples_df, processed_data_location, dataset_name)

    # save_flattened_dataset_to_parquet(flattened_df, processed_data_location, dataset_name)
    save_flattened_dataset_to_gzip(flattened_df, processed_data_location, dataset_name)

    return


# %%
if __name__ == "__main__":
    args = get_args()
    process_dataset(args.dataset_location, args.dataset_name, args.processed_data_location)
