# MAYBE HAVE ONE hyperparameters DICT PER MODEL
batch_parameters = {
    "gap": 10,             # Initial is 10, 5 seems to work better for FFNN, try this also for Transformers
    "total_len": 50,       # Initial is 100 for Transformers, 50 seems to work better for FFNN and OneLayerNN, try this also for Transformers
    "batch_size": 128,      # Initial is 32 for Transformer, 64 for FFNN, but better results even for 128, and even better with 256
}
    
hyperparameters = {
    "epochs": 2,            # Try with more epochs for FFNN
    "dropout": 0.1,         # Initial is 0.3 for Transformer, 0.3 seems to work better for FFNN, 0.1 for One-Layer NN
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "layer_norm_eps": 1e-5,
    "learning_rate": 0.0001,    # 0.0001 for Transformer, 0.0001 seems to work better for FFNN
    "weight_decay": 0.0001,      # Normally 0.0001 for transformers and FFNN
    "n_estimators": 1000,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,     
    "gamma": 0.1,  
}