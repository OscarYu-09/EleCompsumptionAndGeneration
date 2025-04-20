# model_trainer.py
from sklearn.ensemble import GradientBoostingRegressor  # Regression algorithm (decision trees)
from sklearn.model_selection import train_test_split    # Split data into train/test sets
from sklearn.metrics import r2_score, mean_squared_error  # Model evaluation metrics
import pandas as pd  # Load CSV data
import joblib  # Save trained models to .pkl files

class EnhancedDualTrainer:
    def __init__(self):
        # Model configurations (hyperparameters tuned to prevent overfitting)
        self.sunny_model = GradientBoostingRegressor(
            n_estimators=100,  # Number of trees (controls model complexity)
            max_depth=5,        # Max tree depth (limits overfitting)
            learning_rate=0.05, # Shrinkage factor (smaller = more precise)
            verbose=1           # Print training progress
        )
        self.cloudy_model = GradientBoostingRegressor(  # Same config for cloudy days
            n_estimators=100, 
            max_depth=5,
            learning_rate=0.05,
            verbose=1
        )
    
    def _validate_data(self, df):
        """Verify preprocessed data contains required features.
        - df: DataFrame from processed_data.csv
        - Values checked: day_sin, day_cos, etc. (must match preprocessing)
        """
        required_features = ['day_sin', 'day_cos', 'hour_sin', 'hour_cos']
        for feat in required_features:
            assert feat in df.columns, f"Missing the Necessary Features: {feat}"

    def train(self, data_path):
        try:
            # Load preprocessed data
            df = pd.read_csv(data_path, parse_dates=['datetime'])  # From data_preprocessor.py
            self._validate_data(df)  # Ensure feature consistency
            
            # Split data by weather condition (two separate models)
            sunny_df = df[df['is_sunny'] == 1]   # Filter sunny-day records
            cloudy_df = df[df['is_sunny'] == 0]   # Filter cloudy-day records
            
            # Define input features (MUST match solar_predictor.py)
            feature_columns = ['day_sin', 'day_cos', 'hour_sin', 'hour_cos']
            
            # Train sunny-day model
            X_sun = sunny_df[feature_columns]  # Input features (time patterns)
            y_sun = sunny_df['generation']     # Target variable (power output)
            X_train, X_test, y_train, y_test = train_test_split(
                X_sun, y_sun, 
                test_size=0.2,   # 20% data for testing
                random_state=42  # Seed for reproducibility
            )
            self.sunny_model.fit(X_train, y_train)  # Training process
            self._evaluate_model(self.sunny_model, X_test, y_test, "Sunny-Day")
            # Train cloudy-day model (identical workflow)
            X_clo = cloudy_df[feature_columns]
            y_clo = cloudy_df['generation']
            X_train, X_test, y_train, y_test = train_test_split(X_clo, y_clo, test_size=0.2)
            self.cloudy_model.fit(X_train, y_train)
            self._evaluate_model(self.sunny_model, X_test, y_test, "Cloudy-day")
            # Save models to disk (reusable for predictions)
            joblib.dump(self.sunny_model, 'sunny_gbr.pkl')  # File name fixed
            joblib.dump(self.cloudy_model, 'cloudy_gbr.pkl')
            print("\n✅ Model Training Complete")

        except Exception as e:
            print(f"\n❌ Training Fail: {str(e)}")
            raise

    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Calculate model accuracy metrics."""
        y_pred = model.predict(X_test)  # Generate predictions
        print(f"\n=== {model_name}Model Evaluation ===")
        print(f"R² Score: {r2_score(y_test, y_pred):.3f}")  # 1.0 = perfect fit
        print(f"Mean Squared Error(MSE): {mean_squared_error(y_test, y_pred):.3f}")  # Lower is better

if __name__ == "__main__":
    trainer = EnhancedDualTrainer()
    trainer.train('/Users/oscaryu/Downloads/School/materials/ScienceFair/generation/processed_data.csv')  # Output from data_preprocessor.py