class LabelEncoder:
    def __init__(self):
        self.label_mapping = {}
        self.reverse_mapping = {}

    def fit(self, data):
        unique_labels = set(data)
        for i, label in enumerate(unique_labels):
            self.label_mapping[label] = i
            self.reverse_mapping[i] = label

    def transform(self, data):
        return [self.label_mapping[label] for label in data]

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, encoded_data):
        return [self.reverse_mapping[i] for i in encoded_data]
