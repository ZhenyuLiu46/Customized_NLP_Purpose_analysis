import data_load


class CleanedPurposeDataset():
    def __init__(self, all_data):
        self.all_data = all_data

    def clean_data(self):
        return self.all_data


inputData = data_load.PurposeDataset().all_data
cleanInput = CleanedPurposeDataset(inputData).all_data
# Test function and print
for data in cleanInput:
    if not data:
        continue
    else:
        print(data)
