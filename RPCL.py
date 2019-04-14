
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import copy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


k=4
ratio=0.95


# In[38]:


def distance(X,pre_center,competitive=True):
    result=np.empty(len(X))
    pre_center_number=np.empty(k)
    for i in range(len(X)):
        tmp=[]
        for j in range(k):
            tmp.append(((X[i]-pre_center[j])**2).sum())
            result[i]=tmp.index(min(tmp))
    for i in range(k):
        pre_center[i]=X[result==i].mean(0)
        pre_center_number[i]=np.sum(result==i)
    print(pre_center_number)
    if competitive:
        minimum_distance=float('inf')
        minimum_index=np.array([0,0])
        for i in range(k):
            for j in range(k):
                if(j<=i):
                    continue
                if(((pre_center[i]-pre_center[j])**2).sum()<minimum_distance):
                    minimum_distance=((pre_center[i]-pre_center[j])**2).sum()
                    minimum_index[0]=i
                    minimum_index[1]=j
        if(pre_center_number[minimum_index[0]]>pre_center_number[minimum_index[1]]):
            pre_center[minimum_index[1]]=push(pre_center[minimum_index[1]],pre_center[minimum_index[0]])
        else:
            pre_center[minimum_index[0]]=push(pre_center[minimum_index[0]],pre_center[minimum_index[1]])
    return result


# In[39]:


def push(push_center,center):
    return (push_center-center)*(1+ratio)+center


# In[40]:


def generate_dataset(location=np.array([[0,0],[6,0],[3,5]]),n_samples=300,centers=3):
    y=np.empty(n_samples)
    X=np.empty([n_samples,2])
    for i in range(n_samples):
        for j in range(centers):
            tmp=np.random.randint(0,centers)
            X[i]=location[tmp]+np.random.rand(2)*2
            y[i]=tmp
    return X,y


# In[44]:


def competitive_k_means(save_plot=False):
    plt.figure(figsize=(12,12))
    X,y=generate_dataset()
    plt.scatter(X[:,0],X[:,1],c=y,marker='+')
    plt.title("result from the data")
    if save_plot:
        plt.savefig("data.png")
    pre_center=np.empty((k,2))
    for i in range(k):
        pre_center[i]=X[i]
    y_pred=distance(X,pre_center)
    his_y_pred=np.empty(len(X))
    iteration_time=1
    while np.sum(his_y_pred!=y_pred)!=0:
        iteration_time+=1
        his_y_pred=copy.copy(y_pred)
        y_pred=distance(X,pre_center)
        plt.figure()
        plt.scatter(X[:,0],X[:,1],c=y_pred,marker='+')
        plt.scatter(pre_center[:,0],pre_center[:,1],c='r')
        if save_plot:
            plt.savefig("%dinterations.jpg"%iteration_time)
    plt.show()


# In[47]:


competitive_k_means()

