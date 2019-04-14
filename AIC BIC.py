
# coding: utf-8

# In[48]:


import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture


# In[49]:


print(__doc__)


# In[57]:


# # data set 1: 1000-samples each cluster, 2-components, 2-dim
# # Generate random sample, two components
n_samples = 1000
np.random.seed(0)
C = np.array([[0., 1], [1.7, .4]])
print(C)
D=np.array([-6, 3])
print(D)
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

# data set 2: 500-samples each cluster, 5-components, 2-dim
# Generate random sample, two components
# n_samples = 500
# np.random.seed(0)
# C = np.array([[0., 1], [1.7, .4]])
# X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
#           .5 * np.random.randn(n_samples, 2) + np.array([-13, 1]),
#           .5 * np.dot(np.random.randn(n_samples, 2), C) + np.array([6, 3]),
#           .7 * np.dot(np.random.randn(n_samples, 2), C) + np.array([-8, -3])]

# # data set 3: 500 samples each cluster
# # Generate random sample, two components
# n_samples = 500
# np.random.seed(0)
# C = np.array([[0., -0.1], [1.7, .4]])
# X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]


# In[51]:


lowest_bic=np.infty
bic=[]
aic=[]
n_components_range=range(1,7)
cv_types=['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm=mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        aic.append(gmm.aic(X))
        if bic[-1] < lowest_bic:
            lowest_bic=bic[-1]
            best_gmm=gmm
aic=np.array(aic)
bic=np.array(bic)
color_iter=itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange', 'r'])
clf=best_gmm
bars=[]
print(clf)


# In[52]:


plt.figure(figsize=(8,6))
spl=plt.subplot(2,1,1)
for i,(cv_type,color) in enumerate(zip(cv_types,color_iter)):
    xpos=np.array(n_components_range)+ .2*(i-2)
    bars.append(plt.bar(xpos,aic[i*len(n_components_range): (i+1)*len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([aic.min()*1.01 - .01*aic.max(),aic.max()])
plt.title('AIC score per model')
xpos=np.mod(aic.argmin(),len(n_components_range)) + .65 +     .2*np.floor(aic.argmin()/len(n_components_range))
plt.text(xpos,aic.min()*0.97 + .03*aic.max(),'*',fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[53]:


splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X)
for i, (mean,cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    v,w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)
plt.xticks(())
plt.yticks(())
plt.title('Selected GMM:'+ str(clf.covariance_type)+' model,'+str(clf.n_components)+' components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()


# In[37]:


plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +       .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)








# In[ ]:


# data set 1: 200-samples each cluster, 2-components, 10-dim
# Generate random sample, two components
n_samples=500
np.random.seed(0)
X=np.r_[np.random.randn(n_samples,10),
        .7 * np.random.randn(n_samples,10)+np.array([-6,3,3,3,3,3,3,3,3,3]),
        .5 * np.random.randn(n_samples,10)+np.array([-13,1,1,1,1,1,1,1,13,13])]

