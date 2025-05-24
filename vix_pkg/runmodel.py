from copy_train_six_cmf_model import train_six_cmf_models
from etf_predicted_returns import *
from send_orders import send_daily_orders
from calculate_features import calculate_features
import warnings

warnings.simplefilter('ignore')

if __name__ == "__main__":
    calculate_features()
    
    final_preds = train_six_cmf_models("vix_pkg/data/vix_features_calculated.xlsx")
    print("Final shape:", final_preds.shape)
    print(final_preds.head(10))
    
    daily_sizing = get_daily_sizing()

    send_daily_orders(daily_sizing)
    
    
    