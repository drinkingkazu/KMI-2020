import numpy as np

def linear_regression(a,b,num_sample=1000):
    '''
    Generate a fake data sample for linear regression of y=ax+b
    y values are smeared by a normal distribution
    INPUT:
      - a ... float, slope part of a linear equation
      - b ... float, offset part of a linear equation
    '''
    xval = np.zeros(shape=(num_sample),dtype=np.float32)
    yval = np.zeros(shape=(num_sample),dtype=np.float32)
    
    partition = np.concatenate([[i+0.5]*(i) for i in range(10)])
    for i in range(num_sample):
        idx = int(np.random.random() * len(partition))
        xval[i] = np.random.normal(loc=partition[idx],scale=0.5,size=1)
        mean  = a*xval[i] + b
        sigma = np.max([a*xval[i]/5,0.2])
        yval[i] = np.random.normal(loc=mean,scale=sigma,size=1)
    return xval,yval