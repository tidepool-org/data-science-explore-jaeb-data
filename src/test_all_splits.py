import equation_utils
import utils
from pumpsettings import PumpSettings
from test_equations import run_equation_testing

directory_path = "/Users/annaquinlan/Desktop/jaeb-analysis/.reports/phi-uniq_set-3hr_hyst-2020_08_29_23-v0_1_develop-12c5af2/evaluate-equations-2020_10_11_01-v0_1-1635e10/data-processing"
num_splits = 5
group = utils.DemographicSelection.OVERALL

for file_number in range(1, num_splits + 1):
    matching_key = "test_" + str(file_number) + "_" + group.name.lower()
    matching_name = utils.find_matching_file_name(matching_key, ".csv", directory_path)

    # TODO: when Rayhan has the equations, chose the correct one for each split
    jaeb = PumpSettings(
        equation_utils.jaeb_basal_equation,
        equation_utils.jaeb_isf_equation,
        equation_utils.jaeb_icr_equation,
    )

    traditional = PumpSettings(
        equation_utils.traditional_basal_equation,
        equation_utils.traditional_isf_equation,
        equation_utils.traditional_icr_equation,
    )

    # This will output the results to a file
    run_equation_testing(matching_name, jaeb, traditional)

