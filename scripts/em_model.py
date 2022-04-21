#!/usr/bin/env python3
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm, Normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_gaussian_mixture(gm, X, data=True, fit=True, posterior=False, h=0.02, pad=0.25):
  """Plots a sklearn GaussianMixture model fitted to 1D or 2D data.

  Arg: gm (GaussianMixture model: model fitted to X).
  Arg: X (np.array: training data).
  Arg: data (boolean: display training data as as histogram (1D) or scatterplot (2D), default True)
  Arg: fit (boolean: display overall model fit, default True)
  Arg: posterior (boolean: display per-component posterior probability, default False)
  Arg: h (float: step size in grid along which function values are plotted, default 0.02)
  Arg: pad (float: outside padding for grid, default 0.25)
  """

  (n,p) = X.shape
  plt.clf()
    
  if p == 1:
    # 1D plot
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad

    xx = np.matrix(np.arange(x_min, x_max, h)).T
    z = gm.score_samples(xx)

    handle, label = [], []
    
    if data:
      dummy1_, dummy2_, tmp = plt.hist(X,density=True,bins=20)
      handle.append(tmp[0])
      label.append('Histogram')
    
    if fit:
      tmp, = plt.plot(xx,np.exp(z))
      handle.append(tmp)
      label.append('Mixture density')

    if posterior:
      prob = gm.predict_proba(xx)
      for i in range(len(gm.weights_)):
        tmp, = plt.plot(xx,prob[:,i])
        handle.append(tmp)
        label.append('Posterior '+str(i+1))

    if data or fit or posterior:
      plt.legend(handle,label)
    
  else:
    # 2D plot
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = -gm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    if posterior:
      prob = gm.predict_proba(np.c_[xx.ravel(), yy.ravel()])
      tmp = plt.contourf(xx, yy, prob[:,np.argmax(gm.weights_)].reshape(xx.shape), levels=15)
      plt.colorbar(tmp, shrink=0.8, extend="both")

    if fit:
      tmp = plt.contour(xx, yy, z, norm=LogNorm(vmin=1.0, vmax=1000.0), 
                       levels=np.logspace(0, 3, 10))
      plt.colorbar(tmp, shrink=0.8, extend="both")
  
    if data:
      plt.scatter(X[:,0], X[:,1], 0.8, c='red')

    plt.axis('square')
    
  plt.axis('tight')
  plt.show()

values = []
labels = []

with open("./out/graph_metrics.csv", "r") as handle:
    header = handle.readline().strip().split(",")
    for line in handle:
        line = line.strip().split(",")
        values.append([float(line[0]), float(line[1])])
        labels.append(int(line[2]))

values = np.array(values)
labels = np.array(labels)

plt.scatter(values[:, 0][labels == 1], values[:, 1][labels == 1], c="b", s=1, label="Antibacterial")
plt.scatter(values[:, 0][labels == 0], values[:, 1][labels == 0], c="r", s=1, label="Not antibacterial")
plt.xlabel("Average connectivity")
plt.ylabel("Average betweenness centrality")
plt.legend()
plt.savefig("./out/conn_vs_centr.png", dpi=300)
plt.clf()

X_trn, X_tst, y_trn, y_tst = train_test_split(values, labels, train_size=0.5)

results = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trn, y_trn)
    preds = knn.predict(X_tst)
    acc = accuracy_score(y_tst, preds)
    results.append((k, acc))

best = sorted(results, key=lambda x: x[1])[-1]
print(f"best: {best}")

knn = KNeighborsClassifier(n_neighbors=best[0])
knn.fit(X_trn, y_trn)
preds = knn.predict(X_tst)
print(classification_report(y_tst, preds, target_names=["Not antibacterial", "Antibacterial"]))


fpr, tpr, _ = metrics.roc_curve(y_tst,  preds)
auc = metrics.roc_auc_score(y_tst, preds)
plt.plot(fpr, tpr, color='k', label="AUC="+str(round(auc, 2)))
plt.plot([0, 1], [0, 1], color='r', linestyle=":", label="x=y")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc=4)
# plt.savefig(os.path.join(save_dir, "auc_curve.png"), dpi=300)
# plt.show()
plt.clf()








