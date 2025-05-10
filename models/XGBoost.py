import xgboost as xgb

class XGBoostModel:
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, objective="reg:squarederror"):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,               
            colsample_bytree=colsample_bytree, 
            gamma=gamma,                       
            random_state=random_state,
            objective="reg:squarederror"
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self, model_path):
        self.model.load_model(model_path)