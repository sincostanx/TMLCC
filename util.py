import pandas as pd
import os

target = 'CO2_working_capacity [mL/g]'

def getMoreFeature(args, train, test, feature):
    if feature == "box-param":
        train_path = os.path.join(args.root, args.train_cifparam_path)
        test_path = os.path.join(args.root, args.test_cifparam_path)
    elif feature == "coulomb-matrix":
        train_path = os.path.join(args.root, args.train_coulomb_path)
        test_path = os.path.join(args.root, args.test_coulomb_path)
        
    train_feature = pd.read_csv(train_path)
    test_feature = pd.read_csv(test_path)
    
    train = pd.concat([train, train_feature], axis=1)
    test = pd.concat([test, test_feature], axis=1)
    
    return train, test

def getCompoundFeature(args, train, test, feature, drop = False):
    if feature == "density":
        train['density'] = train['weight [u]']/train['volume [A^3]']
        test['density'] = test['weight [u]']/test['volume [A^3]']
    elif feature == "pseudo-target":
        THRESHOLD = 200 # in units of mL CO2/g
        train['bins'] = [1 if value > THRESHOLD else 0 for value in train[target]]
    
    return train, test