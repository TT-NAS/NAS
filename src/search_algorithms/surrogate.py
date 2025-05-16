from xgboost import XGBRegressor

class SurrogateModel(XGBRegressor):
    """
    Clase para el modelo subrogado
    """
    def __init__(self, model_path: str = None, **kwargs) -> None:
        """
        Inicializa el modelo de regresión XGBoost
        """
        super().__init__(**kwargs)
        defaults = {
                'n_estimators': 991,
                'learning_rate': 0.0538,
                'max_depth': 13,
                'random_state': 402,
                'objective': 'reg:squarederror',
                'subsample': 0.8824,
                'colsample_bytree': 0.8785,
                'gamma': 0.0553,
                'reg_lambda': 0.0827,
                'reg_alpha': 1.3668
            }

        if model_path is not None:
            self.load_model(model_path)

        else:
            for key, value in defaults.items():
                kwargs.setdefault(key, value)
            super().__init__(**kwargs)
