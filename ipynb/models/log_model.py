import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 14

from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, f1_score, brier_score_loss

pd.set_option('display.float_format', lambda x: '%3f' % x)

def gen_auc_viz(fpr, tpr, threshold):
    
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    plt.figure(figsize=(8,6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.title('Reorder (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show();

def plot_cm(classifier,
             X_var,
             y_var,
             pred_var,
             cmap=plt.cm.Blues,
             title='Confusion matrix'):
    
    plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues)  
    plt.show() 
    
    cm = pd.DataFrame(metrics.confusion_matrix(y_true=y_var, y_pred=pred_var))
    
    tn = cm.iloc[0,0]
    fn = cm.iloc[1,0]
    fp = cm.iloc[0,1]
    tp = cm.iloc[1,1]
    
    acc = (tp+tn) / (tp+tn+fp+fn)
    sen = (tp) / (tp+fn) #summarizes how well the positive class was predicted.
    prec = tp / (tp+fp) #summarizes the fraction of examples assigned the positive class that belong to the positive class
    rec = tp/(tp+fn) #same as sensitivity 
    pp = tp / (tp+fp)
    
    return acc, sen, prec, rec, pp

 def scores_output(y_test, y_pred):

    f1=f1_score(y_test,y_pred)
    brier=brier_score_loss(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    
    print(classification_report(y_test,y_pred))
    print(f"f1:{f1:.5f},auc:{auc:.5f},loss_score:{brier:.5f}")

full_df = pd.read_csv('../../data/full_df.csv')
# train_df = pd.read_csv('../../data/train_df.csv')
# prior_df = pd.read_csv('../../data/prior_df.csv')
# test_df = pd.read_csv('../../data/test_df.csv')


full_df.drop('Unnamed: 0',axis=1,inplace=True)
full_df.head()

#filling in null values - for previous calculations where user didn't do any reorders
full_df.product_ordered_twice_vol.fillna(0,inplace=True)
full_df.product_order_twice_ratio.fillna(0,inplace=True)
full_df.user_unique_reorder_count.fillna(0,inplace=True)
full_df.user_unique_reorder_perc.fillna(0,inplace=True)
full_df.user_total_items_after_first_order.fillna(0,inplace=True)
full_df.user_reorder_ratio.fillna(0,inplace=True)
full_df.isnull().sum()

print(full_df.target.value_counts())
print(full_df.target.value_counts(normalize=True))

sns.heatmap(full_df.corr(),vmin=-1, vmax=1, center=0,
           cmap=sns.diverging_palette(20, 220, n=200))

full_df.drop(['product_ordered_once_vol','product_reordered_vol',
             'user_unique_product_count','user_unique_reorder_count',
             'user_total_items_after_first_order'],axis=1, inplace=True)

sns.heatmap(full_df.corr(),vmin=-1, vmax=1, center=0,
           cmap=sns.diverging_palette(20, 220, n=200))

#Multicolinearity Check and Dropping columns
cor_matrix = full_df.corr().abs()
upper = cor_matrix.where(
    np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
# print(upper)

drop_these = [column for column in upper.columns if any(upper[column] > 0.80)]

full_new = full_df.drop(full_df[drop_these], axis=1)
full_new.head()

#Split data into train/test/validation
np.random.seed(42)
train, validate, test = \
        np.split(full_new.sample(frac=1, random_state=42), 
                 [int(.6*len(full_new)), int(.8*len(full_new))])
#                  [int(.8*len(full_new))])                 

X= train.drop(["target",'eval_set'],axis=1)
y= train['target']

train_mean = X.mean()
train_std = X.std()

X_val = validate.drop(["target",'eval_set'],axis=1)
X_fin = test.drop(["target",'eval_set'],axis=1)
pd.concat([X.mean(),X_val.mean(),X_fin.mean()], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


#scaling all features minus ID using columntransformer
col_names = X_train.drop(['user_id','product_id','order_id'],axis=1)
col_names = col_names.columns
all_cols = X_train.columns
features = X_train[all_cols]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ct = ColumnTransformer([('train_scaler', StandardScaler(), col_names)], remainder='passthrough')

X_train_scaled = ct.fit_transform(features)
X_test_scaled = ct.transform(X_test)

cols_reordered = ['user_product_count', 'user_product_first_order',
         'user_product_last_order', 'user_product_avg_basket_placement',
         'product_ordered_vol', 'product_order_twice_ratio',
         'user_total_products', 'user_unique_product_perc', 'user_total_orders',
         'user_lifetime_days', 'user_avg_days_between_orders',
         'user_max_time_between_orders', 'user_min_time_between_orders',
         'user_avg_cart_size', 'user_product_order_rate',
         'user_product_reorder_rate', 'user_product_last_time_product_ordered',
         'user_id','product_id','order_id']
X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns = cols_reordered) 
X_train_scaled_df
X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns = cols_reordered)
X_test_scaled_df

X_train_scaled_df['user_id'] = X_train_scaled_df.user_id.astype('int')
X_train_scaled_df['order_id'] = X_train_scaled_df.order_id.astype('int')
X_train_scaled_df['product_id'] = X_train_scaled_df.product_id.astype('int')

X_test_scaled_df['user_id'] = X_test_scaled_df.user_id.astype('int')
X_test_scaled_df['order_id'] = X_test_scaled_df.order_id.astype('int')
X_test_scaled_df['product_id'] = X_test_scaled_df.product_id.astype('int')

X_train_scaled_df
X_test_scaled_df

#viz with PCA
from sklearn.decomposition import PCA

X_train_scaled = X_train_scaled_df.iloc[:,:17]
X_test_scaled = X_test_scaled_df.iloc[:,:17]

pca = PCA(n_components=2)
X_train_dim_red = pca.fit_transform(X_train_scaled)
X_test_dim_red = pca.transform(X_test_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
for label, color in zip(set(y_train), ('blue', 'orange')):
    ax.scatter(
        X_train_dim_red[y_train == label, 0],
        X_train_dim_red[y_train == label, 1],
        color=color, label=f'Class {label}',
        alpha=0.1)

#Set final training for clarity 
X_train = X_train_scaled_df
X_test = X_test_scaled_df 

#Baseline Logistic Regression + visuals 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print(logreg.score(X_test, y_test))

reorder_logit_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
gen_auc_viz(fpr, tpr, thresholds)

plot_cm(logreg, X_test, y_test, y_pred)

f1=f1_score(y_test,y_pred)
brier=brier_score_loss(y_test,y_pred)
auc=roc_auc_score(y_test,y_pred)

print(classification_report(y_test,y_pred))
print(f"f1:{f1:.5f},auc:{auc:.5f},loss_score:{brier:.5f}")









