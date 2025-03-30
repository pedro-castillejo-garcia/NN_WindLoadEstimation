# MAYBE HAVE ONE hyperparameters DICT PER MODEL
batch_parameters = {
    "gap": 5,             # Initial is 10, 5 seems to work better for FFNN, try this also for Transformers
    "total_len": 50,       # Initial is 100 for Transformers, 50 seems to work better for FFNN and OneLayerNN, try this also for Transformers
    "batch_size": 64,      # Initial is 32 for Transformer, 64 for FFNN, but better results even for 128, and even better with 256
}
    
hyperparameters = {
    "epochs": 20,                # Try with more epochs for FFNN
    "dropout": 0.3,             # Initial is 0.3 for Transformer, 0.3 seems to work better for FFNN, 0.1 for One-Layer NN
    "d_model": 64,              #This is for Transformer
    "nhead": 4,                 #This is for Transformer
    "num_layers": 2,            #This is for Transformer
    "dim_feedforward": 256,     #This is for Transformer
    "layer_norm_eps": 1e-5,     #This is for Transformer
    "learning_rate": 0.0001,    # 0.0001 for Transformer, 0.0001 for FFNN and OneLayerNN
    "weight_decay": 0.0001,     # Normally 0.0001 for transformers and FFNN
    "n_estimators": 1000,       # This is for XGBoost
    "max_depth": 10,            # This is for XGBoost
    "subsample": 0.8,           # This is for XGBoost
    "colsample_bytree": 0.8,    # This is for XGBoost
    "gamma": 0.1,               # This is for XGBoost
}