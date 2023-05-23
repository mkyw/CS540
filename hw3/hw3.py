from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x 

def get_covariance(dataset):
    # Your implementation goes here!
    dataset = (np.dot(np.transpose(dataset), dataset))/(len(dataset)-1)
    return dataset

def get_eig(S, m):
    # Your implementation goes here!
    n = len(S)
    Lamda, U = eigh(S, subset_by_index=[n-m, n-1])
    Lamda = np.diag(np.flip(Lamda))
    U = np.flip(U, axis=1)
    return Lamda, U 

def get_eig_prop(S, prop):
    # Your implementation goes here!
    n = len(S)
    Lamda, U = eigh(S, subset_by_value=[prop*np.trace(S), np.inf])
    Lamda = np.diag(np.flip(Lamda))
    U = np.flip(U, axis=1)
    return Lamda, U

def project_image(image, U):
    # Your implementation goes here!
    UT = np.transpose(U)
    alpha = np.dot(UT,image)
    projection = np.dot(alpha, UT)
    return projection

def display_image(orig, proj):
    # Your implementation goes here!
    
    orig = np.transpose(np.reshape(orig, (32,32)))
    proj = np.transpose(np.reshape(proj, (32,32)))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.set_title('Original')
    ax2.set_title('Projection')
    
    og = ax1.imshow(X=orig, aspect='equal')
    pr = ax2.imshow(X=proj, aspect='equal')

    fig.colorbar(og, ax=ax1)
    fig.colorbar(pr, ax=ax2)

    plt.show()
   
    pass

x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
display_image(x[0], projection)