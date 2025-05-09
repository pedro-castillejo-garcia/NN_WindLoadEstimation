# MAYBE HAVE ONE hyperparameters DICT PER MODEL
batch_parameters = {
    "gap": 10,             # Initial is 10 ( also the one that works the best)
    "total_len": 100,       # Initial is 100 for Transformers, 500 works better for OneLayerNN
    "batch_size": 64,      # Initial is 32 for Transformer, 64 for FFNN, but better results even for 128, and even better with 256
}                          # batchsize for rfb: gap1, 32, gap10 64, same with cnn

hyperparameters = {
    "epochs": 100,                # Try with more epochs for FFNN
    "dropout": 0.3,             # Initial is 0.3 for Transformer, 0.3 seems to work better for FFNN, 0 for One-Layer NN
    "d_model": 64,              #This is for Transformer
    "nhead": 4,                 #This is for Transformer
    "num_layers": 2,            #This is for Transformer
    "dim_feedforward": 256,     #This is for Transformer
    "layer_norm_eps": 1e-5,     #This is for Transformer
    "learning_rate": 0.05,    # 0.0001 for Transformer, 0.0001 for FFNN and OneLayerNN, rbf: 0.05, cnn: 1e-3
    "weight_decay": 1e-5,     # Normally 0.0001 for transformers and FFNN
    "n_estimators": 1000,       # This is for XGBoost
    "max_depth": 10,            # This is for XGBoost
    "subsample": 0.8,           # This is for XGBoost
    "colsample_bytree": 0.8,    # This is for XGBoost
    "gamma": 0.1,               # This is for XGBoost
    
    "num_channels": [32, 64, 64],               # For TCN
    "kernel_size": 3,                           # For TCN, 3 for CNN
    "kernel_initializer": 'kaiming_uniform',    # For TCN
    "causal": True,                             # For TCN
    "use_skip_connections": False,              # For TCN
    "use_norm": "weight_norm",                  # For TCN
    "activation": "relu",                       # For TCN

    "cnn_filters": 32,                          # For CNN_LSTM
    "lstm_hidden": 64,                          # For LSTM & CNN_LSTM
    "dense_units": 256,                         # For LSTM & CNN_LSTM
    "num_layers_lstm": 2,                        # For LSTM & CNN_LSTM

    "beta_init": 0.1,                           # For rbf
    "num_hidden_neurons": 100                   # For rbf
}
