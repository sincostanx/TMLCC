import os
from option import args
from metrics import LMAE

import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import torch
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer

def showShape(x_train, y_train, x_val, y_val, x_test):
    print("X train shape: ", x_train.shape)
    print("X validation shape: ", x_val.shape)
    print("X test shape: ", x_test.shape)
    print("Y train shape: ", y_train.shape)
    print("Y validation shape: ", y_val.shape)

def getTabnetParam(args):
    trainer_param = dict(
                    seed = args.seed,
                    n_d = 8,
                    n_a = 8,
                    gamma = 1.3,
                    n_independent = 2,
                    n_steps = 1,
                    lambda_sparse = 1e-5,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=1e-2),
                    mask_type='entmax' # "sparsemax"
                )
    
    tabnet_param = dict(
                   n_d = 8,
                   n_a = 8,
                   gamma = 1.3,
                   n_independent = 2,
                   n_steps = 3,
                   lambda_sparse = 1e-5,
                   optimizer_fn=torch.optim.Adam,
                   optimizer_params=dict(lr=2e-2, weight_decay = 5e-4),
                   scheduler_params={"step_size":2, # how to use learning rate scheduler
                                     "gamma":0.9},
                   scheduler_fn=torch.optim.lr_scheduler.StepLR,
                   mask_type='entmax' # "sparsemax"
               )
    return trainer_param, tabnet_param

def trainTabnet(train, test, args):
    checkpoint_path = os.path.join(args.root, args.checkpoint_path)
    
    x = train[features].to_numpy()
    dummy_y = train['bin'].to_numpy()
    y = train[target].to_numpy().reshape(-1, 1)
    x_test = test[features].to_numpy(dtype=np.int64)
    
    train_pred = np.zeros((train.shape[0], 1))
    test_pred = np.zeros((test.shape[0], args.fold))
    
    i = 1
    kfold = KFold(n_splits = args.fold, shuffle=True, random_state=args.seed)
    
    for train_index, val_index in kfold.split(x, dummy_y):
        print("################# START FOLD ", i, "#################")
        
        pretrainer_name = "Pretrainer-it" + str(0) + '-' + str(i)
        model_name = "Tabnet-it" + str(0) + '-' + str(i)
        
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]
        showShape(x_train, y_train, x_val, y_val, x_test)
        
        if args.mode == "train":
            trainer_param, tabnet_param = getTabnetParam(args)
            
            if args.pretrainer:
                unsupervised_model = TabNetPretrainer(**trainer_param)
                unsupervised_model.fit(
                    X_train=x_train,
                    eval_set=[x_val],
                    pretraining_ratio=0.8,
                    batch_size=1024, virtual_batch_size=128,
                    max_epochs = 200,
                )
                unsupervised_model.save_model(os.path.join(checkpoint_path, pretrainer_name))
            else: unsupervised_model = None
            
            model = TabNetRegressor(**tabnet_param)
            model.fit(
                x_train,y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                eval_name=['train', 'valid'],
                eval_metric=[LMAE],
                max_epochs=200 , patience=50,
                batch_size=1024, virtual_batch_size=128,
                from_unsupervised=unsupervised_model,
                drop_last=False
            )
            
            model.save_model(os.path.join(checkpoint_path, model_name))
            
            print(model.feature_importances_)
        
        if args.mode == "eval":
            model = TabNetRegressor()
            model.load_model(os.path.join(checkpoint_path, model_name + ".zip"))
        
        train_pred[val_index, :] = model.predict(x_val).reshape(-1, 1)
        test_pred[:, i-1] = model.predict(x_test)[0]
        print(test_pred[:, i-1])
        
        i+=1
        
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    print(test_pred)
    print(test_pred_mean)
    print("LMAE score : ", np.log(mean_absolute_error(y, train_pred)))
    
    return train_pred, test_pred_mean

def execute(args):
    ###################################### For reproducibility #########################################
    
    np.random.seed(args.seed)
    
    ###################################### Read dataset #########################################
    train_path = os.path.join(args.root, args.train_path)
    test_path = os.path.join(args.root, args.test_path)
    train_cifparam_path = os.path.join(args.root, args.train_cifparam_path)
    test_cifparam_path = os.path.join(args.root, args.test_cifparam_path)
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_cifparam = pd.read_csv(train_cifparam_path)
    test_cifparam = pd.read_csv(test_cifparam_path)
    
    train.pop('MOFname')
    test.pop('MOFname')
    train = pd.concat([train, train_cifparam], axis=1)
    test = pd.concat([test, test_cifparam], axis=1)
    
    # print(train.columns)
    
    ###################################### Feature Engineering #########################################
    global target, features
    target = 'CO2_working_capacity [mL/g]'
    
    THRESHOLD = 200 # in units of mL CO2/g
    train['density'] = train['weight [u]']/train['volume [A^3]']
    test['density'] = train['weight [u]']/train['volume [A^3]']
    
    train['bin'] = [1 if value > THRESHOLD else 0 for value in train[target]]
    
    # print(test.columns)
    
    ###################################### Handle missing data #########################################
    
    #drop 0, -1 surface area (temporary measure)
    train.drop(train[train['surface_area [m^2/g]'] == -1].index, inplace = True)
    train.drop(train[train['surface_area [m^2/g]'] == 0].index, inplace = True)
    train.drop(train[train['void_fraction'] == -1].index, inplace = True)
    train.drop(train[train['void_fraction'] == 0].index, inplace = True)
    train.drop(train[train['void_volume [cm^3/g]'] == 0].index, inplace = True)
    
    # drop permanent measure
    train.drop(train[train['CO2/N2_selectivity'] == 0].index, inplace = True)
    train.drop(train[train['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]'] == np.inf].index, inplace = True)
    train.drop(train[train['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]'].isna()].index, inplace = True)
    
    #replace inf by nan for imputer
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    #categorical encoding (convert nan to -1) ----> subjected to change later
    str_col = ["crystal_sys", "functional_groups", "topology"]
    # str_col = ["functional_groups", "topology"]
    print(type(train), type(test))
    for i in str_col:
        print(i)
        train[i] = train[i].astype("category").cat.codes
        test[i] = test[i].astype("category").cat.codes
    
    print(train[train.isna().sum(axis=1)!=0])
    
    not_required_transform = ['bin']
    features = [col for col in train.columns if col not in [target] + not_required_transform]
    
    pipe = Pipeline([
            # ('imputer', KNNImputer(n_neighbors=5)),
            ("scaler", QuantileTransformer(n_quantiles=256, output_distribution='normal')),
            ])
    train[features] = pipe.fit_transform(train[features])
    test[features] = pipe.transform(test[features])
    
    print(train[features].columns)

    ###################################### Train/Eval models #########################################
    trainpred_path = os.path.join(args.root, args.trainpred_path)
    testpred_path = os.path.join(args.root, args.testpred_path)
    pred_name = 'baseline-tabnet' + str(0) + '.npy'
    
    train_pred, test_pred = trainTabnet(train, test, args)
    
    if not os.path.isdir(trainpred_path): os.makedirs(trainpred_path)
    if not os.path.isdir(testpred_path): os.makedirs(testpred_path)
        
    np.save(os.path.join(trainpred_path, pred_name), train_pred)
    np.save(os.path.join(testpred_path, pred_name), test_pred)

if __name__ == "__main__":
    
    execute(args)
