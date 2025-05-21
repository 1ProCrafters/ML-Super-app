import base64
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import seaborn as sns
import streamlit as st
import xgboost as xgb

from graphviz import Digraph

from dowhy import CausalModel
from dowhy.causal_estimators import CausalEstimator
from dowhy.causal_refuters.data_subset_refuter import DataSubsetRefuter

from prophet import Prophet

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

from upload_data_page import upload_data
from feature_engineering import feature_engineering, display_data_preview, display_handle_missing_values, display_process_currency_percentage
from explore_data import explore_data, display_boxplot, display_binary_distribution, feature_comparison_graph_page
from regression import evaluate_model_page, display_model_performance_comparison, prepare_data, create_models, fit_models, evaluate_models, plot_model_performance
from regression import display_select_target_features_and_train, display_feature_importance, display_prediction_vs_actual, display_residuals_plot
from advance_data_analysis import advanced_data_analysis, perform_classification, perform_clustering, perform_dimensionality_reduction
from time_series_analysis import time_series_analysis, visualize_time_series_data, display_acf_pacf, fit_arima_model
from causation_page import causality_page
from decision_tree_page import decision_tree_page


import advance_data_analysis
import causation_page
import decision_tree_page
import regression
import upload_data_page
import create_model_page