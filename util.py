import pandas as pd
import os

target = 'CO2_working_capacity [mL/g]'

def loadFeature(args, train, test, feature):
    if feature == "overview":
        train_path = os.path.join(args.root, args.train_path)
        test_path = os.path.join(args.root, args.test_path)
    elif feature == "box-param":
        train_path = os.path.join(args.root, args.train_cifparam_path)
        test_path = os.path.join(args.root, args.test_cifparam_path)
    elif feature == "metal-center":
        train_path = os.path.join(args.root, args.train_metal_path)
        test_path = os.path.join(args.root, args.test_metal_path)
    elif feature == "coulomb-matrix":
        train_path = os.path.join(args.root, args.train_coulomb_path)
        test_path = os.path.join(args.root, args.test_coulomb_path)
        
        # train_feature = pd.read_csv(train_path)
        # train = pd.concat([train, train_feature.iloc[:, 0]], axis=1)
        
        # test_feature = pd.read_csv(test_path)
        # test = pd.concat([test, test_feature.iloc[:, 0]], axis=1)
        
        # return train, test
        
    elif feature == "gsa-restored":
        # train.pop('surface_area [m^2/g]')
        # test.pop('surface_area [m^2/g]')
        train['surface_label'] = 0
        train['void_label'] = 0
        train_path = os.path.join(args.root, args.train_gsa_path)
        df = pd.read_csv(train_path)
        limit = 1
        # print(df.columns)
        # print(df.describe())
        for i in range(6703):
            # print(train.loc[df.iloc[i, 0]-1, 'surface_area [m^2/g]'], train.loc[df.iloc[i, 0]-1, 'void_volume [cm^3/g]'])
            train.loc[df.iloc[i, 0]-1, 'surface_area [m^2/g]'] = df.iloc[i, 2]
            # if train.loc[df.iloc[i, 0]-1, 'void_volume [cm^3/g]'] <= 0:
            train.loc[df.iloc[i, 0]-1, 'void_volume [cm^3/g]'] = int(df.iloc[i, 1]) / 100000.
            train.loc[df.iloc[i, 0]-1, 'void_label'] = 1
            train.loc[df.iloc[i, 0]-1, 'surface_label'] = 1
            
            # print(train.loc[df.iloc[i, 0]-1])
        
        test['surface_label'] = 0
        test['void_label'] = 0
        
        return train, test
        # test_path = os.path.join(args.root, args.test_gsa_path)
        
    if train_path != "null":
        train_feature = pd.read_csv(train_path)
        train = pd.concat([train, train_feature], axis=1)
    
    if test_path != "null":
        test_feature = pd.read_csv(test_path)
        test = pd.concat([test, test_feature], axis=1)
    
    print(">>>>>> Done loading ", feature)
    
    return train, test

def getCompoundFeature(args, train, test, feature, drop = False, str_col = []):
    if feature == "density":
        train['density'] = train['weight [u]']/train['volume [A^3]']
        test['density'] = test['weight [u]']/test['volume [A^3]']
        
        # train.drop(train[train['weight [u]'] == -1].index, inplace = True)
        # test.drop(test[test['weight [u]'] == -1].index, inplace = True)
        
        # train.drop(train[train['volume [A^3]'] == -1].index, inplace = True)
        # test.drop(test[test['volume [A^3]'] == -1].index, inplace = True)
        
    elif feature == "pseudo-target":
        THRESHOLD = 200 # in units of mL CO2/g
        train['bins'] = [1 if value > THRESHOLD else 0 for value in train[target]]
        
    elif feature == "void-volume":
        K = 1.660539
        
        row = (train['void_fraction'] > 0) & (train['void_volume [cm^3/g]'] <= 0)
        train.loc[row, 'void_volume [cm^3/g]'] = (train.loc[row, 'void_fraction'] * train.loc[row, 'volume [A^3]']) / (K * train.loc[row, 'weight [u]'])
        print("Restored {} void volume in training set".format(row.values.sum()))
        
        row = (test['void_fraction'] > 0) & (test['void_volume [cm^3/g]'] <= 0)
        test.loc[row, 'void_volume [cm^3/g]'] = (test.loc[row, 'void_fraction'] * test.loc[row, 'volume [A^3]']) / (K * test.loc[row, 'weight [u]'])
        print("Restored {} void volume in testing set".format(row.values.sum()))
        
    
    elif feature == "void-fraction":
        K = 1.660539
        
        row = (train['void_fraction'] <= 0) & (train['void_volume [cm^3/g]'] > 0)
        train.loc[row, 'void_fraction'] = (K * train.loc[row, 'void_volume [cm^3/g]'] * train.loc[row, 'weight [u]']) / train.loc[row, 'volume [A^3]']
        print("Restored {} void fraction in training set".format(row.values.sum()))
        
        row = (test['void_fraction'] <= 0) & (test['void_volume [cm^3/g]'] > 0)
        test.loc[row, 'void_fraction'] = (K * test.loc[row, 'void_volume [cm^3/g]'] * test.loc[row, 'weight [u]']) / test.loc[row, 'volume [A^3]']
        print("Restored {} void fraction in training set".format(row.values.sum()))
    
    elif feature == "ohe":
        enc = OneHotEncoder(drop = 'first')
    
        temp = enc.fit_transform(train[str_col])
        train_ohe = pd.DataFrame(temp.toarray())
        train = pd.concat([train.reset_index(drop=True), train_ohe], axis=1)
        train = train.drop(str_col, axis=1)
        
        temp = enc.transform(test[str_col])
        test_ohe = pd.DataFrame(temp.toarray())
        test = pd.concat([test.reset_index(drop=True), test_ohe], axis=1)
        test = test.drop(str_col, axis=1)
    
    elif feature == "label":
        for i in str_col:
            train[i] = train[i].astype("category").cat.codes
            test[i] = test[i].astype("category").cat.codes
    
    print(">>>>>> Done calculating ", feature)
    
    return train, test