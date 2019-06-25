import config
import data_load
import data_clean


class DataPrepPickle:
    def __init__(self):
        self.prepare_pickle()

    def prepare_pickle(self):
        input_data = data_load.PurposeDataset().all_data  # load data
        # saves to pickle file, problbly should save seperately?
        data_clean.CleanedPurposeDataset(
            input_data).clean_data(input_data)
        print("data_pickle.pkl saved")
