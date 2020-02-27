from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from sklearn.cluster import DBSCAN
import matplotlib
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

def save_state(blob, prefix='./snapshot'):
    # Output file name
    filename = '%s-%d.ckpt' % (prefix, blob.iteration)
    # Save parameters
    # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
    # 2) network weight
    torch.save({
        'global_step': blob.iteration,
        'optimizer': blob.optimizer.state_dict(),
        'state_dict': blob.net.state_dict()
        }, filename)
    return filename

def restore_state(blob):
    # Open a file in read-binary mode
    with open(blob.weight_file, 'rb') as f:
        # torch interprets the file, then we can access using string keys
        checkpoint = torch.load(f)
        # load network weights
        blob.net.load_state_dict(checkpoint['state_dict'], strict=False)
        # if optimizer is provided, load the state of the optimizer
        if blob.optimizer is not None:
            blob.optimizer.load_state_dict(checkpoint['optimizer'])
        # load iteration count
        blob.iteration = checkpoint['global_step']

def plot_dataset(dataset,num_image_per_class=10):
    import numpy as np
    num_class = 0
    classes = []
    if hasattr(dataset,'classes'):
        classes=dataset.classes
        num_class=len(classes)
    else: #brute force
        for data,label in dataset:
            if label in classes: continue
            classes.append(label)
        num_class=len(classes)
    
    shape = dataset[0][0].shape
    big_image = np.zeros(shape=[3,shape[1]*num_class,shape[2]*num_image_per_class],dtype=np.float32)
    
    finish_count_per_class=[0]*num_class
    for data,label in dataset:
        if finish_count_per_class[label] >= num_image_per_class: continue
        img_ctr = finish_count_per_class[label]
        big_image[:,shape[1]*label:shape[1]*(label+1),shape[2]*img_ctr:shape[2]*(img_ctr+1)]=data
        finish_count_per_class[label] += 1
        if np.sum(finish_count_per_class) == num_class*num_image_per_class: break
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,8),facecolor='w')
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelleft=False,labelbottom=False)
    plt.imshow(np.transpose(big_image,(1,2,0)))
    for c in range(len(classes)):
        plt.text(big_image.shape[1]+shape[1]*0.5,shape[2]*(c+0.6),str(classes[c]),fontsize=16)
    plt.show()

# Plot a confusion matrix
def plot_confusion_matrix(label,prediction,class_names):
    """
    Args: label ... 1D array of true label value, the length = sample size
          prediction ... 1D array of predictions, the length = sample size
          class_names ... 1D array of string label for classification targets, the length = number of categories
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value  = np.max([np.max(np.unique(label)),np.max(np.unique(label))])
    assert max_value < num_labels
    mat,_,_,im = ax.hist2d(label,prediction,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=16)
    ax.set_yticklabels(class_names,fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, str(mat[i, j]),
                    ha="center", va="center", fontsize=16,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    plt.show()

class ImagePadder(object):
    
    def __init__(self,padsize=20,randomize=False):
        self._pads=padsize
        self._randomize=randomize
        
    def __call__(self,img):
        import numpy as np
        img=np.array(img)
        new_shape = list(img.shape)
        new_shape[0] += self._pads
        new_shape[1] += self._pads
        
        new_img = np.zeros(shape=new_shape,dtype=img.dtype)
        pad_vertical   = int(self._pads / 2.)
        pad_horizontal = int(self._pads / 2.)
        if self._randomize:
            pad_vertical   = int(np.random.uniform() * self._pads)
            pad_horizontal = int(np.random.uniform() * self._pads)
        
        new_img[pad_vertical:pad_vertical+img.shape[0],pad_horizontal:pad_horizontal+img.shape[1]] = img
        return new_img

# Compute moving average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Decorative progress bar
def progress_bar(count, total, message=''):
    """
    Args: count .... int/float, current progress counter
          total .... int/float, total counter
          message .. string, appended after the progress bar
    """
    from IPython.display import HTML, display,clear_output
    return HTML("""
        <progress 
            value='{count}'
            max='{total}',
            style='width: 30%'
        >
            {count}
        </progress> {frac}% {message}
    """.format(count=count,total=total,frac=int(float(count)/float(total)*100.),message=message))

# Memory usage print function
def print_memory(msg=''):
    max_allocated = round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)
    allocated = round_decimals(torch.cuda.memory_allocated()/1.e9, 3)
    max_cached = round_decimals(torch.cuda.max_memory_cached()/1.e9, 3)
    cached = round_decimals(torch.cuda.memory_cached()/1.e9, 3)
    print(max_allocated, allocated, max_cached, cached, msg)

# Function to plot softmax score for N-class classification (good for N<10)
def plot_softmax(label,softmax):
    import numpy as np
    import matplotlib.pyplot as plt
    num_class   = len(softmax[0])
    unit_angle  = 2*np.pi/num_class
    xs = np.array([ np.sin(unit_angle*i) for i in range(num_class+1)])
    ys = np.array([ np.cos(unit_angle*i) for i in range(num_class+1)])
    fig,axis=plt.subplots(figsize=(10,10),facecolor='w')
    plt.plot(xs,ys)#,linestyle='',marker='o',markersize=20)
    for d in range(num_class):
        plt.text(xs[d]*1.1-0.04,ys[d]*1.1-0.04,str(d),fontsize=24)
    plt.xlim(-1.3,1.3)
    plt.ylim(-1.3,1.3)
    
    xs=xs[0:10]
    ys=ys[0:10]
    for d in range(num_class):
        idx=np.where(label==d)
        scores=softmax[idx]
        xpos=[np.sum(xs * s) for s in scores]
        ypos=[np.sum(ys * s) for s in scores]
        plt.plot(xpos,ypos,linestyle='',marker='o',markersize=10,alpha=0.5)
    axis.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelleft=False,labelbottom=False)
    plt.show()



# Dumb class to organize output csv file
class CSVData:

    def __init__(self,fout):
        self.name  = fout
        self._fout = None
        self._str  = None
        self._dict = {}

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):
        if self._str is None:
            self._fout=open(self.name,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()

def forward(blob,train=True):
    """
       Args: blob should have attributes, net, criterion, softmax, data, label
       
       Returns: a dictionary of predicted labels, softmax, loss, and accuracy
    """
    with torch.set_grad_enabled(train):
        # Prediction
        data = blob.data.cuda()
        prediction = blob.net(data)
        # Training
        loss,acc=-1,-1
        if blob.label is not None:
            label = blob.label.cuda() #label = torch.stack([ torch.as_tensor(l) for l in np.hstack(label) ])
            loss = blob.criterion(prediction,label)
        blob.loss = loss
        
        softmax    = blob.softmax(prediction).cpu().detach().numpy()
        prediction = torch.argmax(prediction,dim=-1)
        accuracy   = (prediction == label).sum().item() / float(prediction.nelement())        
        prediction = prediction.cpu().detach().numpy()
        
        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : loss.cpu().detach().item(),
                'accuracy'   : accuracy}

def backward(blob):
    blob.optimizer.zero_grad()  # Reset gradients accumulation
    blob.loss.backward()
    blob.optimizer.step()

def train_loop(blob,train_loader,num_iteration):
    # Set the network to training mode
    blob.net.train()
    # Let's record the loss at each iteration and return
    train_loss=[]
    # Progress bar decoration
    from IPython.display import display
    progress=display(progress_bar(0,num_iteration),display_id=True)
    # Loop over data samples and into the network forward function
    while blob.iteration < num_iteration:
        for i,data in enumerate(train_loader):
            if blob.iteration >= num_iteration:
                break
            blob.iteration += 1
            # data and label
            if isinstance(data,dict):
                blob.data, blob.label = data['data'], data['label']
            else:
                blob.data, blob.label = data
            if not isinstance(blob.data,torch.Tensor):
                blob.data=torch.Tensor(blob.data).unsqueeze(1).cuda()
                blob.label=torch.Tensor(blob.label).long()
            # call forward
            res = forward(blob,True)
            # Recird loss
            train_loss.append(res['loss'])
            # once in a while, report
            message='Iteration: %d Loss: %.2f Accuracy: %.2f' % (blob.iteration,res['loss'],res['accuracy'])
            if blob.iteration%10 == 0:
                progress.update(progress_bar(blob.iteration,num_iteration,message=message))
            backward(blob)
    progress.update(progress_bar(num_iteration,num_iteration,message=message))
    return np.array(train_loss)

def plot_loss(loss,num_average=30,iterations_per_epoch=None):
    import numpy as np
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,8),facecolor='w')
    iterations=np.array(range(len(loss))).astype(np.float32)
    if iterations_per_epoch is not None:
        iterations = iterations / iterations_per_epoch
    
    plt.plot(iterations,loss,marker="",linewidth=2,color='blue',label='loss (raw)')
    plt.plot(moving_average(iterations,num_average),moving_average(loss,num_average),
             marker="",linewidth=2,color='red',label='rolling mean')
    ax.set_xlabel('Iterations' if iterations_per_epoch is None else 'Epoch',fontsize=20)
    ax.set_ylabel('Loss',fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid(True,which='both')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.show()


def inference_loop(blob,dataloader,local_data_dir='./'):
    import numpy as np
    # set the network to test (non-train) mode
    blob.net.eval()
    # create the result holder
    accuracy, label, prediction, softmax = [], [], [], []
    confusion_matrix = np.zeros([10,10],dtype=np.int32)
    for i,data in enumerate(dataloader):
        if isinstance(data,dict):
            blob.data, blob.label = data['data'], data['label']
        else:
            blob.data, blob.label = data
        if not isinstance(blob.data,torch.Tensor):
            blob.data=torch.Tensor(blob.data).unsqueeze(1).cuda()
            blob.label=torch.Tensor(blob.label).long()
        res = forward(blob,True)
        accuracy.append(res['accuracy'])
        prediction.append(res['prediction'])
        label.append(blob.label)
        softmax.append(res['softmax'])
    # organize the return values
    accuracy   = np.hstack(accuracy)
    prediction = np.hstack(prediction)
    label      = np.hstack(label)
    softmax    = np.vstack(softmax)
    return accuracy, label, prediction, softmax
