import data_load
import re  # for replace special character


class CleanedPurposeDataset():
    def __init__(self, _all_data):
        self.all_data = [[]]
        self.X = []
        self.y = []
        self.clean_data(_all_data)  # load X,y,all_data

    def clean_data(self, _all_data):
        cleaned_data = [[]]
        text = []
        label = []
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
            text.append(data[0])
            label.append(data[1])
        del cleaned_data[0]  # delete first empty element
        self.X = text
        self.y = label
        self.all_data = cleaned_data


# Test function and print

'''inputData = data_load.PurposeDataset().all_data
cleaned_data = CleanedPurposeDataset(inputData).all_data
cleaned_data_X = CleanedPurposeDataset(inputData).X
cleaned_data_y = CleanedPurposeDataset(inputData).y


for data in cleaned_data:
    print(data)

for data1 in cleaned_data_X:
    print(data1)

for data2 in cleaned_data_y:
    print(data2)
'''
