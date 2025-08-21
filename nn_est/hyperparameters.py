batch_parameters = {
    "gap": 1,           
    "total_len": 100,      
    "batch_size": 64,    
}

hyperparameters = {
    "epochs": 100,              
    "dropout": 0.3,            
    "d_model": 64,              #This is for Transformer
    "nhead": 4,                 #This is for Transformer
    "num_layers": 2,            #This is for Transformer
    "dim_feedforward": 256,     #This is for Transformer
    "layer_norm_eps": 1e-5,     #This is for Transformer
    "learning_rate": 0.0001,    
    "weight_decay": 0.0001,     
    "n_estimators": 1000,     
    "max_depth": 10,            
    "subsample": 0.8,          
    "colsample_bytree": 0.8,    
    "gamma": 0.1,               
    
    "num_channels": [32, 64, 64],               # For TCN
    "kernel_size": 5,                           # For TCN
    "kernel_initializer": 'kaiming_uniform',    # For TCN
    "causal": True,                             # For TCN
    "use_skip_connections": False,              # For TCN
    "use_norm": "weight_norm",                  # For TCN
    "activation": "relu",                       # For TCN

    "cnn_filters": 32,                          # For CNN_LSTM
    "lstm_hidden": 64,                          # For LSTM & CNN_LSTM
    "dense_units": 256,                         # For LSTM & CNN_LSTM
    "num_layers_lstm": 2                        # For LSTM & CNN_LSTM
}