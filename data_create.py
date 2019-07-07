import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# takes equal amounts of samples from each category
# returns train test data set
def create_sample(sample_amount):
    if sample_amount > 4000:
        raise ValueError('cannot sample more then 4000 data points.')
    data = pd.read_csv('data/wrangled_train.csv', index_col=0)
    sampled_text = []
    sampled_id = []
    sampled_tag = []
    for tag in list(data.Tag.unique()):
        sampled = data[data['Tag'] == tag].sample(sample_amount, random_state=42)
        sampled_text_ = list(sampled['Text'])
        sampled_id_ = list(sampled['ID'])
        sampled_tag_ = list(sampled['Tag_num'])

        for item in sampled_text_:
            sampled_text.append(item)
        for item in sampled_id_:
            sampled_id.append(item)
        for item in sampled_tag_:
            sampled_tag.append(item)

    with open('sampled_data.txt', 'w') as outfile:
        for sample_index in range(len(sampled_id)):
            json.dump({sampled_id[sample_index]: [sampled_text[sample_index], sampled_tag[sample_index]]}, outfile)
            outfile.write('\n')

    with open('sampled_data.txt', 'r') as infile:
        data = [json.loads(i) for i in infile if type(list(json.loads(i).values())[0][0]) == str]
    X_train, X_test, y_train, y_test = train_test_split([list(i.values())[0][0] for i in data],
                                                        [list(i.values())[0][1] for i in data], test_size=0.20,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test

def feature_vectors(X_train, X_test, y_train, y_test):
    split = len(X_train)
    all_doc = ''
    for doc in X_train + X_test:
        all_doc = all_doc + ' ' + doc
    feature_extraction = TfidfVectorizer(all_doc)
    data = feature_extraction.fit_transform(X_train + X_test)
    labels = y_train + y_test
    X_train = data[:split]
    X_test = data[split:]
    y_train = labels[:split]
    y_test = labels[split:]
    y_train = list(map(str, y_train))
    y_test = list(map(str, y_test))

    return X_train, X_test, y_train, y_test

