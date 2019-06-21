import data_load
import re  # for replace special character


class CleanedPurposeDataset():
    def __init__(self, _all_data):
        self.all_data = self.clean_data(_all_data)

    def clean_data(self, _all_data):
        cleaned_data = [[]]
        for data in _all_data:
            # data[0]: text, data[1]: label
            data[0] = re.sub(r'\W+', ' ', data[0])  # remove special chars
            data[0] = re.sub(r'[0-9]', ' ', data[0])  # remove digits
            # remove single char word
            data[0] = re.sub(r'\b[a-zA-Z]\b', '', data[0])
            data[0] = re.sub(r'\s+', ' ', data[0]).strip()  # remove whitespace
            if not data[0]:  # skip empty strings
                continue
            cleaned_data.append(data)
        del cleaned_data[0]  # delete first empty element
        return cleaned_data


# Test function and print
inputData = data_load.PurposeDataset().all_data
cleanInput = CleanedPurposeDataset(inputData).all_data

for data in cleanInput:
    print(data)
