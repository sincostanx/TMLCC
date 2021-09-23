import os
from option import args
from metrics import LMAE

import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer

from xgboost import XGBRegressor

from util import getMoreFeature, getCompoundFeature

def showShape(x_train, y_train, x_val, y_val, x_test):
    print("X train shape: ", x_train.shape)
    print("X validation shape: ", x_val.shape)
    print("X test shape: ", x_test.shape)
    print("Y train shape: ", y_train.shape)
    print("Y validation shape: ", y_val.shape)

def getXGBParam():
    return {
          'tree_method' : 'gpu_hist', 
          'learning_rate' : 0.01,
          'n_estimators' : 50000,
          'colsample_bytree' : 0.3,
          'subsample' : 0.75,
          'reg_alpha' : 19,
          'reg_lambda' : 19,
          'max_depth' : 5, 
          'predictor' : 'gpu_predictor'
    }

def getTabnetParam(args):
    trainer_param = dict(
                    seed = args.seed,
                    n_d = 8,
                    n_a = 8,
                    gamma = 1.3,
                    n_independent = 2,
                    n_steps = 2,
                    lambda_sparse = 1e-3,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=1e-2),
                    mask_type='entmax' # "sparsemax"
                )
    
    tabnet_param = dict(
                   n_d = 16,
                   n_a = 16,
                   gamma = 1.3,
                   n_independent = 2,
                   n_steps = 3,
                   lambda_sparse = 1e-5,
                   optimizer_fn=torch.optim.Adam,
                   optimizer_params=dict(lr=1e-2, weight_decay = 1e-5),
                   scheduler_params={"step_size":3, # how to use learning rate scheduler
                                     "gamma":0.9},
                   scheduler_fn=torch.optim.lr_scheduler.StepLR,
                   mask_type='entmax' # "sparsemax"
               )
    return trainer_param, tabnet_param

def trainXGB(train, test, args):
    checkpoint_path = os.path.join(args.root, args.checkpoint_path)
    
    x = train[features].to_numpy()
    dummy_y = train['bins'].to_numpy()
    y = train[target].to_numpy().reshape(-1, 1)
    x_test = test[features].to_numpy()
    print(x_test)
    
    train_pred = np.zeros((train.shape[0], 1))
    test_pred = np.zeros((test.shape[0], args.fold))
    
    i = 1
    kfold = KFold(n_splits = args.fold, shuffle=True, random_state=args.seed)
    
    for train_index, val_index in kfold.split(x, dummy_y):
        print("################# START FOLD ", i, "#################")
        
        model_name = "XGB-it" + str(0) + '-' + str(i) + '.json'
        
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]
        showShape(x_train, y_train, x_val, y_val, x_test)
        
        if args.mode == "train":
            xgb_params = getXGBParam()
            model = XGBRegressor(**xgb_params)
            model.fit(x_train, y_train,
                    eval_set=[(x_val, y_val)],
                    early_stopping_rounds=200,
                    verbose=500, eval_metric = 'rmse')
            model.save_model(os.path.join(checkpoint_path, model_name))
        
        if args.mode == "eval":
            model = XGBRegressor()
            model.load_model(os.path.join(checkpoint_path, model_name))
        
        train_pred[val_index, :] = model.predict(x_val).reshape(-1, 1)
        test_pred[:, i-1] = model.predict(x_test)
        print(test_pred[:, i-1])
        
        i+=1
        
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    print(test_pred)
    print(test_pred_mean)
    print("LMAE score : ", np.log(mean_absolute_error(y, train_pred)))
    
    return train_pred, test_pred_mean

def trainTabnet(train, test, args):
    checkpoint_path = os.path.join(args.root, args.checkpoint_path)
    
    x = train[features].to_numpy()
    dummy_y = train['bins'].to_numpy()
    y = train[target].to_numpy().reshape(-1, 1)
    x_test = test[features].to_numpy()
    print(x_test)
    
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
                    batch_size=256, virtual_batch_size=128,
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
                max_epochs=300 , patience=50,
                batch_size=1024, virtual_batch_size=128,
                from_unsupervised=unsupervised_model,
                drop_last=False
            )
            
            model.save_model(os.path.join(checkpoint_path, model_name))
        
        if args.mode == "eval":
            model = TabNetRegressor()
            model.load_model(os.path.join(checkpoint_path, model_name + ".zip"))
        
        train_pred[val_index, :] = model.predict(x_val).reshape(-1, 1)
        test_pred[:, i-1] = np.squeeze(model.predict(x_test))
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
    
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    
    train.pop('MOFname')
    test.pop('MOFname')
    print(train.info())
    print(test.info())
    
    if args.boxParam: train, test = getMoreFeature(args, train, test, "box-param")
    if args.coulomb: train, test = getMoreFeature(args, train, test, "coulomb-matrix")
    
    ###################################### Feature Engineering #########################################
    global target, features
    target = 'CO2_working_capacity [mL/g]'
    
    # print(test.columns)
    
    ###################################### Handle missing data #########################################
    
    #drop 0, -1 surface area (temporary measure)
    train.drop(train[train['surface_area [m^2/g]'] == -1].index, inplace = True)
    train.drop(train[train['surface_area [m^2/g]'] == 0].index, inplace = True)
    train.drop(train[train['void_fraction'] == -1].index, inplace = True)
    train.drop(train[train['void_fraction'] == 0].index, inplace = True)
    train.drop(train[train['void_volume [cm^3/g]'] == 0].index, inplace = True)
    
    # drop (permanent measure)
    train.drop(train[train['CO2/N2_selectivity'] == 0].index, inplace = True)
    train.drop(train[train['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]'] == np.inf].index, inplace = True)
    train.drop(train[train['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]'].isna()].index, inplace = True)
    
    # problem_col = ['CO2/N2_selectivity', 'heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]']
    # train[problem_col].replace([-1, 0, np.inf, -np.inf], np.nan, inplace=True)
    # test[problem_col].replace([-1, 0, np.inf, -np.inf], np.nan, inplace=True)
    
    train, test = getCompoundFeature(args, train, test, "density")
    if args.pseudo_target: train, test = getCompoundFeature(args, train, test, "pseudo-target")
    
    
    str_col = ["functional_groups", "topology"]
    if args.boxParam: str_col += ["crystal_sys"]
    
    not_required_transform = [target, "bins"] if args.pseudo_target else [target]
    
    print(not_required_transform)
    transform_col = [col for col in train.columns if col not in not_required_transform + str_col]
    
    pipe = Pipeline([
            # ('imputer', KNNImputer(n_neighbors=5)),
            ("scaler", QuantileTransformer(n_quantiles=256, output_distribution='normal')),
            # ("pca", PCA())
            ])
    # print(transform_col)
    train[transform_col] = pipe.fit_transform(train[transform_col])
    test[transform_col] = pipe.transform(test[transform_col])
    
    print("Numeric transform done")
    
    if args.encoding == "ohe":
        #one hot encodeing
        enc = OneHotEncoder(drop = 'first')
    
        temp = enc.fit_transform(train[str_col])
        train_ohe = pd.DataFrame(temp.toarray())
        train = pd.concat([train.reset_index(drop=True), train_ohe], axis=1)
        train = train.drop(str_col, axis=1)
        
        temp = enc.transform(test[str_col])
        test_ohe = pd.DataFrame(temp.toarray())
        test = pd.concat([test.reset_index(drop=True), test_ohe], axis=1)
        test = test.drop(str_col, axis=1)
        
    elif args.encoding == "label":
        for i in str_col:
            train[i] = train[i].astype("category").cat.codes
            test[i] = test[i].astype("category").cat.codes

    print("String transform done")
    # ###################################### Train/Eval models #########################################
    
    
    features = [col for col in train.columns if col not in not_required_transform]
    
    # print(test[features].info())
    
    trainpred_path = os.path.join(args.root, args.trainpred_path)
    testpred_path = os.path.join(args.root, args.testpred_path)
    pred_name = 'baseline' + str(0) + '.npy'
    
    if args.model == "tabnet":
        train_pred, test_pred = trainTabnet(train, test, args)
    elif args.model == "xgb":
        train_pred, test_pred = trainXGB(train, test, args)
    
    if not os.path.isdir(trainpred_path): os.makedirs(trainpred_path)
    if not os.path.isdir(testpred_path): os.makedirs(testpred_path)
        
    np.save(os.path.join(trainpred_path, pred_name), train_pred)
    np.save(os.path.join(testpred_path, pred_name), test_pred)

if __name__ == "__main__":
    
    execute(args)
