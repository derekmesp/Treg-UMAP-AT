import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import logging

logging.basicConfig(level=logging.INFO)


def regression_pipeline(df, target, features, splits=5, max_depth=None):
    """
    Performs a regression analysis using a Random Forest model on the provided DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data for regression analysis.
    target : str
        The name of the target variable (dependent variable) in the DataFrame.
    features : list of str
        A list of feature names (independent variables) to be used in the regression model.
    splits : int, optional
        The number of splits for K-Fold cross-validation. Default is 5.
    max_depth : int or None, optional
        The maximum depth of the Random Forest trees. If None, nodes are expanded until all leaves are pure. Default is None.
    Returns
    -------
    tuple
        A tuple containing the fitted regression model, a DataFrame of feature importances, and a list of cleaned feature names.
    """

    model_df = df[features + [target]].dropna().copy()
    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['Age']),
            ('cat', TargetEncoder(cv=5, random_state=42),
             ['Tissue', 'Population'])
        ]
    )

    reg_model = Pipeline([
        ('preprocess', preprocess),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_split=15,  # Requires at least 15 samples to attempt a split
            min_samples_leaf=5,     # Ensures final leaf nodes have at least 5 samples
            random_state=42,
            n_jobs=-1
        ))
    ])

    cv_strategy = KFold(n_splits=splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(reg_model, X, y, cv=cv_strategy, scoring='r2')

    logging.info(f"Cross-validation R² scores: {cv_scores}")
    logging.info(
        f"Mean R² score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    reg_model.fit(X_train, y_train)
    y_pred_train = reg_model.predict(X_train)
    y_pred_test = reg_model.predict(X_test)

    logging.info(f"Training R² score: {r2_score(y_train, y_pred_train):.3f}")
    logging.info(f"Test R² score: {r2_score(y_test, y_pred_test):.3f}")

    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_train, y=residuals_train,
                    color='blue', alpha=0.6, label='Train Data')
    sns.scatterplot(x=y_pred_test, y=residuals_test,
                    color='crimson', alpha=0.7, label='Test Data')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Residual Plot: Predicted RTL vs. Residuals (Random Forest Model)')
    plt.xlabel('Predicted Relative Telomere Length (RTL)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    try:
        fitted_preprocess = reg_model.named_steps['preprocess']
        raw_names = fitted_preprocess.get_feature_names_out()

        clean_feature_names = []
        for name in raw_names:
            if name.startswith('num__'):
                clean_feature_names.append(name.replace('num__', ''))
            elif name.startswith('cat__'):
                clean_feature_names.append(
                    name.replace('cat__', '') + '_encoded')
            else:
                clean_feature_names.append(name)
    except AttributeError:
        logging.error(
            "Error occurred while extracting feature names from the preprocessing step.")
        clean_feature_names = ['Age', 'Tissue_encoded', 'Population_encoded']

    importances = reg_model.named_steps['regressor'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': clean_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    logging.info("\nFeature Importances:\n" + importance_df.to_string())

    return reg_model, importance_df, clean_feature_names


def clean_encoder_map(reg_model):
    """
    Extracts and returns a clean mapping of original categorical values to their corresponding encoded values from the
    TargetEncoder used in the regression pipeline.

    Parameters
    ----------
    reg_model : sklearn.pipeline.Pipeline
        The fitted regression pipeline containing the preprocessing step with TargetEncoder.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original categorical values and their corresponding encoded values for each feature.
    """
    try:
        preprocess_step = reg_model.named_steps['preprocess']
        target_encoder = preprocess_step.named_transformers_['cat']
    except AttributeError as e:
        logging.error(
            f"Error occurred while extracting encoder information: {e}")

    features = ['Tissue', 'Population']

    for i, feature in enumerate(features):
        cats = target_encoder.categories_[i]
        encs = target_encoder.encodings_[i]

        feature_map = pd.DataFrame({
            'Original Category': cats,
            'Encoded Value (RTL Mean)': encs
        }).sort_values('Encoded Value (RTL Mean)').reset_index(drop=True)

        logging.info(f"\n=== {feature.upper()} COMPLETE REFERENCE MAP ===")
        logging.info(feature_map.to_string(index=False))

    return feature_map
