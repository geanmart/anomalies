# %%
!pip install pyod   

# %%
import umap
import umap.plot

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
pd.set_option('display.max_columns', 500)

# %%
data =np.load("13_fraud.npz", allow_pickle=True)

# %%
X, y = data['X'], data['y']

# %%
X_data = pd.DataFrame(X).copy()
X_data.columns = [f"feature_{k}" for k in X_data.columns]
X_data["gt"] = pd.Series(y)

# %%
# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def corrigir_formatos(df):
    # Cria uma cópia do DataFrame para evitar modificar o original
    df_corrigido = df.copy()

    for coluna in df_corrigido.columns:
        try:
            # Verifica se a coluna é do tipo float
            if np.issubdtype(df_corrigido[coluna].dtype, np.floating):
                # Verifica se todos os valores são 0.0 ou 1.0
                if df_corrigido[coluna].isin([0.0, 1.0]).all():
                    df_corrigido[coluna] = df_corrigido[coluna].astype(np.int32)  # Converte para int32
                    logging.info(f'Coluna "{coluna}" convertida para int32.')
                else:
                    df_corrigido[coluna] = df_corrigido[coluna].astype(np.float32)  # Deixa como float32
                    logging.info(f'Coluna "{coluna}" mantida como float32.')
                    
            # Verifica se é um inteiro e mantém como int32
            elif np.issubdtype(df_corrigido[coluna].dtype, np.integer):
                df_corrigido[coluna] = df_corrigido[coluna].astype(np.int32)
                logging.info(f'Coluna "{coluna}" convertida para int32.')
                
            # Mantém colunas do tipo Object ou String no formato original
            elif df_corrigido[coluna].dtype == 'object':
                logging.info(f'Coluna "{coluna}" mantida no formato original (object).')
                
        except Exception as e:
            logging.error(f'Erro ao processar a coluna "{coluna}": {e}')

    return df_corrigido


# %%
X_data = corrigir_formatos(X_data)


# %%
def infer_distribution_type(df, column_name):
    col = df[column_name]
    
    # Checa se é um número inteiro ou decimal
    if pd.api.types.is_integer_dtype(col):
        unique_values = col.nunique()
        
        # Definimos um limite arbitrário de 20 valores únicos
        if unique_values < 20:
            return "discrete"
        else:
            # Verifica se é uma sequência uniforme
            diffs = col.sort_values().diff().dropna().unique()
            if len(diffs) == 1:
                return "discrete"
            else:
                return "continous"
    
    elif pd.api.types.is_float_dtype(col):
        return "continous"
    
    else:
        return "non numeric"


# %%
X_data.shape

# %%
mapper = umap.UMAP(n_neighbors=100).fit(X)
umap.plot.points(mapper, labels=y)

# %%


# %%
import sys
import os
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
	os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.inne import INNE
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD

# from pyod.models.dif import DIF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
# from pyod.models.suod import SUOD
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from pyod.models.kpca import KPCA
# from pyod.models.lunar import LUNAR

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# %%
# Example: Training an ECOD detector
from pyod.models.ecod import ECOD
clf = ECOD()
clf.fit(X_train)
y_train_scores = clf.decision_scores_  # Outlier scores for training data
y_test_scores = clf.decision_function(X_test)  # Outlier scores for test data


# %%
full_train = pd.concat([pd.Series(y_train_scores),pd.Series(y_train)],axis=1)

# %%
full_train.columns = ["score","gt"]

# %%
full_train["gt"].value_counts()

# %%
sns.boxplot(data=full_train, x= "score",hue="gt")

# %%
round(0.001798,5)

# %%
pd.Series(y_train).value_counts(normalize=True)

# %%
# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define the number of inliers and outliers
n_samples = X_train.shape[0]
outliers_fraction = 0.0018
clusters_separation = [0]

n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)

# %%
# Train all models

# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
				 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
				 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
				 LOF(n_neighbors=50)]


random_state = 42
# Define nine outlier detection tools to be compared
classifiers = {
	'Angle-based Outlier Detector (ABOD)':
		ABOD(contamination=outliers_fraction),
	'K Nearest Neighbors (KNN)': KNN(
		contamination=outliers_fraction),
	'Average KNN': KNN(method='mean',
					   contamination=outliers_fraction),
	'Median KNN': KNN(method='median',
					  contamination=outliers_fraction),
	'Local Outlier Factor (LOF)':
		LOF(n_neighbors=35, contamination=outliers_fraction),

	'Isolation Forest': IForest(contamination=outliers_fraction,
								random_state=random_state),
	# 'Deep Isolation Forest (DIF)': DIF(contamination=outliers_fraction,
									#    random_state=random_state),
	'INNE': INNE(
		max_samples=2, contamination=outliers_fraction,
		random_state=random_state,
	),

	'Locally Selective Combination (LSCP)': LSCP(
		detector_list, contamination=outliers_fraction,
		random_state=random_state),
	# 'Feature Bagging':
	# 	FeatureBagging(LOF(n_neighbors=35),
	# 				   contamination=outliers_fraction,
	# 				   random_state=random_state),
	# 'SUOD': SUOD(contamination=outliers_fraction),

	'Minimum Covariance Determinant (MCD)': MCD(
		contamination=outliers_fraction, random_state=random_state),

	'Principal Component Analysis (PCA)': PCA(
		contamination=outliers_fraction, random_state=random_state),
	'KPCA': KPCA(
		contamination=outliers_fraction),

	'Probabilistic Mixture Modeling (GMM)': GMM(contamination=outliers_fraction,
												random_state=random_state),

	'LMDD': LMDD(contamination=outliers_fraction,
				 random_state=random_state),

	'Histogram-based Outlier Detection (HBOS)': HBOS(
		contamination=outliers_fraction),

	'Copula-base Outlier Detection (COPOD)': COPOD(
		contamination=outliers_fraction),

	'ECDF-baseD Outlier Detection (ECOD)': ECOD(
		contamination=outliers_fraction),
	'Kernel Density Functions (KDE)': KDE(contamination=outliers_fraction),

	'QMCD': QMCD(
		contamination=outliers_fraction),

	'Sampling': Sampling(
		contamination=outliers_fraction),

	# 'LUNAR': LUNAR(),

	'Cluster-based Local Outlier Factor (CBLOF)':
		CBLOF(contamination=outliers_fraction,
			  check_estimator=False, random_state=random_state),

	'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
}


# %%
# Fit the model
ground_truth = y_train
for i, (clf_name, clf) in enumerate(classifiers.items()):
    print(i + 1, 'fitting', clf_name)
    # fit the data and tag outliers
    clf.fit(X_train)
    scores_pred = clf.decision_function(X_test) * -1
    y_pred = clf.predict(X_test)
    threshold = percentile(scores_pred, 100 * outliers_fraction)
    n_errors = (y_pred != y_test).sum()
    percent_erros = n_errors/y_test.sum()
    print(n_errors,percent_erros)

# %%
clf = IForest(contamination=outliers_fraction,
								random_state=123456)

# %%
clf.fit(X_train)
scores_pred = clf.decision_function(X_test) * -1
y_pred = clf.predict(X_test)
threshold = percentile(scores_pred, 100 * outliers_fraction)
n_errors = (y_pred != y_test).sum()
percent_erros = n_errors/len(y_test)

# %%
results = pd.concat([pd.Series(y_pred),pd.Series(y_test)],axis=1)
results.columns = ["pred_anom","anom"]

# %%
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# %%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
clf_inne = INNE(
		max_samples=8, contamination=outliers_fraction,
		random_state=123456,
	)

clf_inne.fit(X_train)
scores_pred = clf_inne.decision_function(X_test) * -1
y_pred = clf_inne.predict(X_test)
threshold = percentile(scores_pred, 100 * outliers_fraction)
n_errors = (y_pred != y_test).sum()
percent_erros = n_errors/len(y_test)

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)

# %%
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
from pyod.models.xgbod import XGBOD
from pyod.utils.data import evaluate_print


# %%
# train XGBOD detector
clf_name = 'XGBOD'
clf = XGBOD(random_state=42)
clf.fit(X_train, y_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

# %%
%pwd

# %%


# %%



