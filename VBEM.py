
# coding: utf-8

# In[25]:


import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture


# In[26]:


print(__doc__)


# In[27]:


#n_samples=500
#np.random.seed(0)
#X=np.r_[np.random.randn(n_samples,10),
#        .7 * np.random.randn(n_samples,10) + np.array([-6,3,3,3,3,3,3,3,3,3]),
#        .5 * np.random.randn(n_samples,10) + np.array([-13,1,1,1,1,1,1,1,13,13])]

n_samples = 250
np.random.seed(0)
C = np.array([[0., -0.51], [1.7, .4]])
X = np.r_[ np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

#n_samples = 500
#np.random.seed(0)
#C = np.array([[0., 1], [1.7, .4]])
#X = np.r_[.7 * np.dot(np.random.randn(n_samples, 2), C) + np.array([3, -1]),
#          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
#          .5 * np.random.randn(n_samples, 2) + np.array([3, 3]),
#          .5 * np.dot(np.random.randn(n_samples, 2), C) + np.array([-1, 3]),
#          .7 * np.dot(np.random.randn(n_samples, 2), C) + np.array([-6, -1])]


# In[28]:


def plot_results(X, Y_, means, covariances, index, title):
    splot=plt.subplot(2,1,1+index)
    for i,(mean,covar,color) in enumerate(zip(means,covariances,color_iter)):
        v, w = linalg.eigh(covar)
        v=2. * np.sqrt(2.) * np.sqrt(v)
        u=w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i,0],X[Y_ ==i,1],3.8,color=color,alpha=0.3)
        angle=np.arctan(u[1] / u[0])
        angle=180. * angle / np.pi
        ell=mpl.patches.Ellipse(mean,v[0],v[1],180. + angle,color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# In[29]:


def plot_bars(model, n_components, index, model_name):
    plt.subplot(2,1,index)
    plt.bar(range(n_components),model.weights_)
    if index == 2:
        plt.xlabel('index of each components')
    if index == 1:
        plt.ylabel('weigths of each components')
    plt.title(str(model_name))


# In[30]:


n_components = 7
color_iter=itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
cv_types=['spherical','tied','diag','full']
vGMM=mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full')
vGMM.fit(X)
gmm=mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')
plot_results(X, vGMM.predict(X), vGMM.means_, vGMM.covariances_, 1, 'Bayesian Gaussian Mixture')
plt.show()
plot_bars(model=gmm, n_components=n_components, index=1, model_name='GMM')
plot_bars(model=vGMM, n_components=n_components, index=2, model_name="GMM with VBEM")
plt.show()

