import pandas as pd
import logging
import joblib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained entities...')

# Import pretrained entities
model_xgb = joblib.load("trained_entities/model.pkl")
onehot_encoder = joblib.load("trained_entities/onehot_encoder.pkl")
catboost_encoder = joblib.load("trained_entities/catboost_encoder.pkl")
feature_columns = joblib.load("trained_entities/feature_columns.pkl")

# Define optimal threshold
model_th = 0.97
logger.info('Pretrained entities imported successfully...')

# Make prediction
def make_prediction(dt, path_to_file):

    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (model_xgb.predict_proba(dt)[:, 1] > model_th) * 1
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission

# Save information on top features
def json_features(model, feature_columns, path_to_file):
    
    feat_importances = pd.Series(model.feature_importances_, index=feature_columns)
    top5 = {str(k): float(v) for k, v in feat_importances.nlargest(5).to_dict().items()}

    with open(path_to_file, "w", encoding="utf-8") as json_file:
        json.dump(top5, json_file, indent=4)

    logger.info("Файл с топ-5 признаками сохранен")

    return path_to_file

# Plot predicted densities
def plot_density(y_pred, path_to_file):

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.histplot(y_pred, kde=True, stat="density", color="darkcyan", bins=50, alpha=0.6, ax=ax[0])
    ax[0].axvline(x=0.97, color="indigo", linestyle="--", linewidth=1.5, label="best threshold")
    ax[0].set_title("Распределение предсказанных вероятностей")
    ax[0].set_xlabel("Probability")
    ax[0].set_ylabel("Density")
    ax[0].set_xlim(0, 1)
    ax[0].grid(False)
    ax[0].legend()
    
    sns.kdeplot(y_pred, bw_adjust=0.5, color="teal", linewidth=2, fill=True, alpha=0.1, ax=ax[1])
    ax[1].set_yscale("log")
    ax[1].axvline(x=0.97, color="indigo", linestyle="--", linewidth=1.5, label="best threshold")
    ax[1].set_title("Распределение предсказанных вероятностей \n(log scale Y)")
    ax[1].set_xlabel("Probability")
    ax[1].set_ylabel("Log density")
    ax[1].set_xlim(0, 1)
    ax[1].grid(False)
    ax[1].legend()
    
    plt.savefig(path_to_file, dpi=300)
    plt.close()

    logger.info("Файл с распределением предсказанных вероятностей сохранен")

    return path_to_file