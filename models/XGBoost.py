import xgboost as xgb

class XGBoostModel:
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            learning_rate=learning_rate, 
            random_state=random_state
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
