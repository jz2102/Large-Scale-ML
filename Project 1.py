import os
import numpy as np
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# additional imports you may find useful for this assignment
from tqdm import tqdm
from scipy.special import softmax

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label

        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cross-entropy loss of the classifier
#
# x         examples          (d)
# y         labels            (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    u = W@x
    softmax = np.exp(u)/np.sum(np.exp(u), axis=0)
    return np.dot((-y),np.log(softmax))+gamma*np.sum(W**2)/2
# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    u = W@x
    softmax = np.exp(u)/np.sum(np.exp(u), axis=0)
    return np.outer(softmax-y, x)+gamma*W
# test that the function multinomial_logreg_grad_i is indeed the gradient of multinomial_logreg_loss_i
def test_gradient():
    # TODO students should implement this in Part 1
    x = np.array([1, 2, 3])
    y = np.array([1,0,0])
    W = np.ones((3,3))
    gradient = multinomial_logreg_grad_i(x,y,0.0001,W)
    V = np.random.random((3,3))
    ita = 0.0000000001
    print(np.trace(V.T@gradient), (multinomial_logreg_loss_i(x,y,0.0001,W+ita*V)-multinomial_logreg_loss_i(x,y,0.0001,W))/ita)

# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    correct_pred = 0
    c,n=Ys.shape
    u = W@Xs
    softmax = np.exp(u)/np.sum(np.exp(u))
    max_index = np.argmax(softmax,axis=0)
    for i in range(n):
        if Ys[:,i][max_index[i]]==1:
            correct_pred+=1
    return 1-correct_pred/n

# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_total_grad(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    # (d,n) = Xs.shape
    # acc = W * 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_grad_i(Xs[:,i], Ys[:,i], gamma, W)
    # return acc / n;

    (d, n) = Xs.shape
    u = W@Xs
    softmax = np.exp(u)/np.sum(np.exp(u), axis=0)

    return np.matmul(softmax-Ys, Xs.T)/n+gamma*W




# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_total_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    u = W@Xs
    softmax = np.exp(u)/np.sum(np.exp(u), axis=0)
    return np.sum((-Ys)*np.log(softmax))+gamma*np.sum(W**2)/2

    # (d,n) = Xs.shape
    # acc = W * 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_loss_i(Xs[:,i], Ys[:,i], gamma, W)
    # return acc / n;


# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    parameters_list = [W0]
    for i in range(1,num_iters+1):
        W0 = W0 - alpha*multinomial_logreg_total_grad(Xs, Ys, gamma, W0)
        if i%monitor_freq ==0:
            parameters_list.append(W0)
    return parameters_list

# estimate the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# nsamples  number of samples to use for the estimation
#
# returns   the estimated model error when sampling with replacement
def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
    # TODO students should implement this
    n = Xs.shape[1]
    sample_index = np.random.choice(n,nsamples)
    Xs = Xs[:,sample_index]
    Ys = Ys[:,sample_index]
    return multinomial_logreg_error(Xs, Ys, W)

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    c = Ys_tr.shape[0]
    d = Xs_tr.shape[0]
    # Part 1 Testing with definition of limit
    #test_gradient()

    # Part 2 - Time GD with for loops
    #para_list = gradient_descent(Xs_tr,Ys_tr,0.0001,np.random.random((c,d)),1,10,10)

    #Part 3 - Time GD without for loops
    para_list = gradient_descent(Xs_tr,Ys_tr,0.0001,np.random.random((c,d)),1,1000,10)
    # # Part 4 - generate plots

    error_tr = []
    error_te = []
    loss_tr = []
    loss_te = []
    x = []
    error_tr_s_100 = []
    error_te_s_100 = []
    error_tr_s_1000 = []
    error_te_s_1000 = []

    # training error plot
    for i in range(len(para_list)):
        x.append(i * 10)
    for i in tqdm(range(len(para_list))):
        error_tr.append(multinomial_logreg_error(Xs_tr, Ys_tr, para_list[i]))
    for i in tqdm(range(len(para_list))):
        error_te.append(multinomial_logreg_error(Xs_te, Ys_te, para_list[i]))
    # for i in range(len(para_list)):
    #     loss_tr.append(multinomial_logreg_total_loss(Xs_tr, Ys_tr, 0.0001, para_list[i]))
    #     loss_te.append(multinomial_logreg_total_loss(Xs_te, Ys_te, 0.0001, para_list[i]))
    # Estimating with subsamples
    for i in tqdm(range(len(para_list))):
        error_tr_s_1000.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, para_list[i],1000))
    for i in tqdm(range(len(para_list))):
        error_te_s_1000.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, para_list[i],1000))
    for i in tqdm(range(len(para_list))):
        error_tr_s_100.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, para_list[i],100))
    for i in tqdm(range(len(para_list))):
        error_te_s_100.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, para_list[i],100))
    pyplot.plot(x,error_tr, label = "full")
    pyplot.plot(x,error_tr_s_100, "--", label = "sample size 100")
    pyplot.plot(x, error_tr_s_1000, "--",label = "sample size 1000")
    pyplot.title("Training Error Rate")
    pyplot.xlabel("iterations")
    pyplot.ylabel("Training Error Rate")
    pyplot.legend()
    pyplot.savefig("project1_err_tr.png")
    pyplot.close()

    pyplot.plot(x,error_te, label = "full")
    pyplot.plot(x, error_te_s_100, "--",label = "sample size 100")
    pyplot.plot(x, error_te_s_1000, "--",label = "sample size 1000")
    pyplot.title("Test Error Rate")
    pyplot.xlabel("iterations")
    pyplot.ylabel("Test Error Rate")
    pyplot.legend()
    pyplot.savefig("project1_err_te.png")
    pyplot.close()

    # pyplot.plot(x,loss_tr)
    # pyplot.title("Training Loss")
    # pyplot.xlabel("iterations")
    # pyplot.ylabel("Training Loss")
    # pyplot.savefig("project1_loss_tr.png")
    # pyplot.close()
    #
    # pyplot.plot(x,loss_te)
    # pyplot.title("Testing Loss")
    # pyplot.xlabel("iterations")
    # pyplot.ylabel("Test Loss")
    # pyplot.savefig("project1_loss_te.png")

    #print(multinomial_logreg_error(Xs_te, Ys_te, para_list[-1]))