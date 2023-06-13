from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import roc_curve, auc
# Import models from scikit learn module:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import lightgbm as lgb

# libraries others models 
import xgboost as xgb
import os

def plot_and_save_roc_curve(y_true, y_scores, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=1, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    # Create the directory if it doesn't exist
    if not os.path.exists('artifact'):
        os.makedirs('artifact')

    path = "artifact/" + title.replace(" ", "_") + ".png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Create the directory if it doesn't exist
    if not os.path.exists('artifact'):
        os.makedirs('artifact')

    # Save the figure
    path = "artifact/" + title.replace(" ", "_") + ".png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()



def train_and_log_logistic_regression_baseline(
   X_train, y_train, X_test, y_test, random_state, solver='lbfgs', max_iter=100, C=1.0, penalty='l2', run_name="Baseline"
):
    with mlflow.start_run(run_name=run_name):
        log_reg = LogisticRegression(solver=solver, max_iter=max_iter, C=C, penalty=penalty, random_state=random_state)
        log_reg.fit(
            X_train, y_train.ravel()
        )  # Use ravel() to avoid the DataConversionWarning

        mlflow.log_param("Model", run_name)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)

        train_preds = log_reg.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        mlflow.log_metric("train_accuracy", train_accuracy)

        train_probs = log_reg.predict_proba(X_train)[:, 1]
        train_roc_auc = roc_auc_score(y_train, train_probs)
        mlflow.log_metric("train_roc_auc", train_roc_auc)

        mlflow.sklearn.log_model(log_reg, "logistic_regression")

        plot_and_save_confusion_matrix(y_train, train_preds, "Train Confusion Matrix")
        plot_and_save_roc_curve(y_train, train_probs, "Train ROC Curve")

        val_probs = log_reg.predict_proba(X_test)[:, 1]
        val_roc_auc = roc_auc_score(y_test, val_probs)
        mlflow.log_metric("val_roc_auc", val_roc_auc)

        val_preds = log_reg.predict(X_test)
        val_accuracy = accuracy_score(y_test, val_preds)  # calculate validation accuracy
        mlflow.log_metric("val_accuracy", val_accuracy)

        val_preds = log_reg.predict(X_test)
        plot_and_save_confusion_matrix(y_test, val_preds, "Test Confusion Matrix")
        plot_and_save_roc_curve(y_test, val_probs, "Test ROC Curve")

        print(f"ROC AUC on validation set: {val_roc_auc:.4f}")

def train_and_log_model_rf(X_train, y_train, X_test, y_test, n_estimators, max_depth, random_state, run_name="Baseline"):
    # Start an MLflow run with the specified run_name
    with mlflow.start_run(run_name=run_name):
        # Initialize and train the Random Forest Classifier with the given hyperparameters
        rf_clf = RandomForestClassifier(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         random_state=random_state)
        rf_clf.fit(X_train, y_train.ravel())

        # Log the model's hyperparameters
        mlflow.log_param("Model", run_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Calculate and log the accuracy metric for the training dataset
        train_preds = rf_clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        mlflow.log_metric("train_accuracy", train_accuracy)

        # Calculate and log the ROC AUC metric for the training dataset
        train_probs = rf_clf.predict_proba(X_train)[:, 1]
        train_roc_auc = roc_auc_score(y_train, train_probs)
        mlflow.log_metric("train_roc_auc", train_roc_auc)

        # Log the trained model using mlflow.sklearn.log_model()
        mlflow.sklearn.log_model(rf_clf, "random_forest_classifier")

        # Calculate and log the ROC AUC metric for the validation dataset
        val_probs = rf_clf.predict_proba(X_test)[:, 1]
        val_roc_auc = roc_auc_score(y_test, val_probs)
        mlflow.log_metric("val_roc_auc", val_roc_auc)

        # Calculate and log the accuracy metric for the validation dataset
        val_preds = rf_clf.predict(X_test)
        val_accuracy = accuracy_score(y_test, val_preds)
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Plot and save the confusion matrix for the validation dataset
        plot_and_save_confusion_matrix(y_test, val_preds, "Test Confusion Matrix")

        # Plot and save the ROC curve for the validation dataset
        plot_and_save_roc_curve(y_test, val_probs, "Test ROC Curve")

        # Print the ROC AUC score for the validation dataset
        print(f"ROC AUC en el conjunto de validación: {val_roc_auc:.4f}")

def train_and_log_lightgbm(X_train, y_train, X_test, y_val, params, run_name="LightGBM"):
    with mlflow.start_run(run_name=run_name):
        # Crear el conjunto de datos de LightGBM
        lgb_X_train = lgb.Dataset(X_train, label=y_train)
        lgb_X_test = lgb.Dataset(X_test, label=y_val, reference=lgb_X_train)

        mlflow.log_param("Model", run_name)

        # Entrenar el modelo LightGBM
        lgb_model = lgb.train(params, lgb_X_train, valid_sets=lgb_X_test)

        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)

        # Calcular y registrar la métrica de precisión en el conjunto de entrenamiento
        train_preds = np.round(lgb_model.predict(X_train))
        train_accuracy = accuracy_score(y_train, train_preds)
        mlflow.log_metric("train_accuracy", train_accuracy)

        # Calcular y registrar el ROC AUC para el conjunto de entrenamiento
        train_probs = lgb_model.predict(X_train)
        train_roc_auc = roc_auc_score(y_train, train_probs)
        mlflow.log_metric("train_roc_auc", train_roc_auc)

        # Calcular y registrar la métrica de precisión en el conjunto de validación
        val_preds = np.round(lgb_model.predict(X_test))
        val_accuracy = accuracy_score(y_val, val_preds)
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Calcular y registrar el ROC AUC para el conjunto de validación
        val_probs = lgb_model.predict(X_test)
        val_roc_auc = roc_auc_score(y_val, val_probs)
        mlflow.log_metric("val_roc_auc", val_roc_auc)

        # Registrar el modelo
        mlflow.lightgbm.log_model(lgb_model, "lightgbm")

        # Guardar la matriz de confusión para el conjunto de validación
        plot_and_save_confusion_matrix(y_val, val_preds, "Validation Confusion Matrix")

        # Guardar la curva ROC para el conjunto de validación
        plot_and_save_roc_curve(y_val, val_probs, "Validation ROC Curve")

        print(f"ROC AUC en el conjunto de validación: {val_roc_auc:.4f}")        

def train_and_log_xgboost(X_train, y_train, X_test, y_val, params, run_name="XGBoost"):
    
    with mlflow.start_run(run_name=run_name):

        # Crear el conjunto de datos de XGBoost
        xgb_X_train = xgb.DMatrix(X_train, label=y_train)
        xgb_X_test = xgb.DMatrix(X_test, label=y_val)

        # Entrenar el modelo XGBoost
        xgb_model = xgb.train(params, xgb_X_train, evals=[(xgb_X_test, 'validation')])

        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)

        # Registrar el nombre del modelo
        mlflow.log_param("Model", run_name)

        # Calcular y registrar la métrica de precisión en el conjunto de entrenamiento
        train_preds = np.round(xgb_model.predict(xgb_X_train))
        train_accuracy = accuracy_score(y_train, train_preds)
        mlflow.log_metric("train_accuracy", train_accuracy)

        # Calcular y registrar el ROC AUC para el conjunto de entrenamiento
        train_probs = xgb_model.predict(xgb_X_train)
        train_roc_auc = roc_auc_score(y_train, train_probs)
        mlflow.log_metric("train_roc_auc", train_roc_auc)

        # Calcular y registrar la métrica de precisión en el conjunto de validación
        val_preds = np.round(xgb_model.predict(xgb_X_test))
        val_accuracy = accuracy_score(y_val, val_preds)
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Calcular y registrar el ROC AUC para el conjunto de validación
        val_probs = xgb_model.predict(xgb_X_test)
        val_roc_auc = roc_auc_score(y_val, val_probs)
        mlflow.log_metric("val_roc_auc", val_roc_auc)

        # Registrar el modelo
        mlflow.xgboost.log_model(xgb_model, "xgboost")

        # Guardar la matriz de confusión para el conjunto de validación
        plot_and_save_confusion_matrix(y_val, val_preds, "Validation Confusion Matrix")

        # Guardar la curva ROC para el conjunto de validación
        plot_and_save_roc_curve(y_val, val_probs, "Validation ROC Curve")
  
        # save h5 model
        xgb_model.save_model('model.xgb')

        print(f"ROC AUC en el conjunto de validación: {val_roc_auc:.4f}")