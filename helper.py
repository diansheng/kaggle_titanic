import pandas as pd

def import_data():
	df_train = pd.read_csv('data/train.csv', index_col=False, header=0)
	df_test = pd.read_csv('data/test.csv', index_col=False, header=0)
	return df_train, df_test

def _convert_embark_to_digit(row):
    map = {
        'C':0,
        'Q':1,
        'S':2,
    }
    # else, -1
    return map[row['Embarked']] if row['Embarked'] in map else -1
    
def _quantify_df(df):
    pd.to_numeric(df['Pclass']);
    pd.to_numeric(df['Age']);
    pd.to_numeric(df['Fare']);
    pd.to_numeric(df['SibSp']);
    pd.to_numeric(df['Parch']);
    if 'Survived' in df.columns:
        pd.to_numeric(df['Survived']);
    df['gender'] = df.apply(lambda row: 1 if row['Sex'] == 'male' else 0 ,axis=1)
    df['embarked_digits'] = df.apply(_convert_embark_to_digit ,axis=1)
    return df


def import_and_clean_data():
	df_train, df_test = import_data()
	df_train = _quantify_df(df_train)
	df_test = _quantify_df(df_test)
	return df_train, df_test

def handle_missing_values(train_data, test_data):
	from sklearn.preprocessing import Imputer
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	train_data = imp.fit_transform(train_data)
	Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
	test_data = imp.transform(test_data)
	return train_data, test_data