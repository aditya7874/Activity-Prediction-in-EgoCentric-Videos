from __future__ import print_function
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("", one_hot=True)
"""def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

data={}
data[0]=unpickle("cifar-10-batches-py/data_batch_1")
data[1]=unpickle("cifar-10-batches-py/data_batch_2")
data[2]=unpickle("cifar-10-batches-py/data_batch_3")
data[3]=unpickle("cifar-10-batches-py/data_batch_4")
data[4]=unpickle("cifar-10-batches-py/data_batch_5")
data[5]=unpickle("cifar-10-batches-py/test_batch")
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as N
import scipy.io as sc
mat=sc.loadmat('updated_data.mat')
data={}
out={}
j=0
epsilon = 1e-3
#print (epsilon)
"""for i in range(1,73):
    if i%9==1:
        li=[]
        for k in range(i,i+9):
 	    a=mat['trainExamples'+ str(k)]
	    if i>35:
		te=a[0]
	        t=te[0:24576]
	        te[0:24576]=te[24576:]
		te[24576:]=t
   		a[0]=te
	 	li.append(a[0])
	    else:
	    	li.append(a[0])
	    #print ("--------------------")
	    #print (len(a[0][1:]))
	    #print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%")

	#print (li)
	#print (len(a[0]))
	#print (len(li))
        out1=[]
	out2=[]
	output=[]
	for k in range(0,9):
	    if 34>35:	
	   	out1.append(0)
	   	out2.append(1)
	    else:
		out1.append(1)
	   	out2.append(0)
        output.append(out1)
        output.append(out2)
	output=[list(i) for i in zip(*output)]
        out[j]=output
        data[j]=li
	j=j+1
	print (out)
"""
# Parameters
#Shuffling
tr=N.arange(72)
batch=9
no=8
data={}
out={}
j=0
ran=N.random.permutation(tr)
#print (len(ran))
for i in range(no):
    ind=ran[i*batch:(i+1)*batch]
    print (ind)
    out1=[]
    out2=[]
    output=[]
    li=[]
    for k in ind:
	a=mat['trainExamples'+ str(k+1)]
        if k+1>35:
      	    te=a[0]
            t=te[0:24576]
            te[0:24576]=te[24576:]
            te[24576:]=t
            a[0]=te
            li.append(a[0])
        else:
            li.append(a[0])

	#out1=[]
        #out2=[]
        #output=[]
        if k+1>35:
     	    out1.append(0)
            out2.append(1)
        else:
            out1.append(1)
            out2.append(0)
    output.append(out1)
    output.append(out2)
    output=[list(i) for i in zip(*output)]
    out[j]=output
    data[j]=li
    j=j+1
print (out)
#print (data[0])

###############
learning_rate = 0.001
training_epochs = 30000
batch_size = 9

momentum=0.9
# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
#n_hidden_3=256  # 3rd layer number of features
n_input = 49152 # MNIST data input (img shape: 32*32*3)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob=tf.placeholder(tf.float32)
"""def sendvec(labela):
    labelv=[]
    #print(labela)
    for i in range(len(labela)):
        x=[]
        for j in range(10):
            if j==labela[i]:
                x.append(1)
            else:
                x.append(0)
        labelv.append(x)
    return list(labelv)

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)
"""
# Create model

def multilayer_perceptron(x, weights,biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #layer_1=tf.matmul(x,weights['h1'])
    #layer_1=batch_norm_wrapper(layer_1,is_tarining)
    #batch_mean1,batch_var1=tf.nn.moments(layer_1,[0])
    #scale1=tf.Variable(tf.ones([512]))
    #beta1=tf.Variable(tf.zeros([512]))
    #layer_1=tf.nn.batch_normalization(layer_1,batch_mean1,batch_var1,beta['1'],scales['1'],epsilon)   
    layer_1 = tf.nn.relu(layer_1)
    
    layer_1=tf.nn.dropout(layer_1,keep_prob)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2=tf.matmul(layer_1,weights['h2'])
    #layer_2=batch_norm_wrapper(layer_2,is_training)
    #batch_mean2,batch_var2=tf.nn.moments(layer_2,[0])
    #scale2=tf.Variable(tf.ones([512]))
    #beta2=tf.Variable(tf.zeros([512]))
    #layer_2=tf.nn.batch_normalization(layer_2,batch_mean2,batch_var2,beta['2'],scales['2'],epsilon)

    layer_2 = tf.nn.relu(layer_2)
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.relu(layer_3)
    layer_2=tf.nn.dropout(layer_2,keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.nn.relu(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    #'h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    #'b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

"""scales = {
    '1':tf.Variable(tf.ones([512])),
    '2':tf.Variable(tf.ones([512]))
}

beta = {
    '1':tf.Variable(tf.zeros([512])),
    '2':tf.Variable(tf.zeros([512]))
}
"""
#scale1=tf.Variable(tf.ones([512]))
#beta1=tf.Variable(tf.zeros([512]))
#scale2=tf.Variable(tf.ones([512]))
#beta2=tf.Variable(tf.zeros([512]))

# Construct model
pred = multilayer_perceptron(x, weights,biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #a=[[1,2],[2,3],[3,1]]
    #k=tf.random_shuffle(a)
    #print(sess.run(k))
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        total_batch=int("8")
        # Loop over all batches
        for i in range(total_batch):
            #batch_x, batch_y = mnist.train.next_batch(batch_size)

            batch_x=data[i]
            batch_y = out[i]
            #print (len(batch_x))
	    #print (len(batch_y))
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y,keep_prob: 0.5})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", 
                "{:.9f}".format(avg_cost))
	    weights_h1=weights['h1'].eval(sess)
	    print (weights_h1)
    	    weights_h2=weights['h2'].eval(sess)	
	    #print (weights_h2)
     	    weights_out=weights['out'].eval(sess)
    	    biases_b1=biases['b1'].eval(sess)
	    print (biases_b1[0])
     	    biases_b2=biases['b2'].eval(sess)
    	    biases_out=biases['out'].eval(sess)
  	    print (biases_out[0])
	    #beta2=beta['2'].eval(sess)
  	    
	    sc.savemat('weight_h1.mat',mdict={'weight_h1':weights_h1})
	    sc.savemat('weight_h2.mat',mdict={'weight_h2':weights_h2})
	    sc.savemat('weight_out.mat',mdict={'weight_out':weights_out})
	    sc.savemat('biases_b1.mat',mdict={'biases_b1':biases_b1})
	    sc.savemat('biases_b2.mat',mdict={'biases_b2':biases_b2})
	    sc.savemat('biases_out.mat',mdict={'biases_out':biases_out})

    print("Optimization Finished!")
    #a=type(weights['h1'])
    #print (a)
    #weights_h1=weights['h1'].eval(sess)
    #weights_h2=weights['h2'].eval(sess)
    #weights_out=weights['out'].eval(sess)
    #biases_b1=biases['b1'].eval(sess)
    #biases_b2=biases['b2'].eval(sess)
    #biases_out=biases['out'].eval(sess)
    #a=type(weights_h1)
    #print (a)
    #print (array)
    #k=weights_h1
    #b=type(k)
    #print (b)
    #print (N.shape(k))
    #print k.shape
    

#sc.savemat('weight_h1.mat',mdict={'weight_h1':weights_h1})
#sc.savemat('weight_h2.mat',mdict={'weight_h2':weights_h2})
#sc.savemat('weight_out.mat',mdict={'weight_out':weights_out})
#sc.savemat('biases_b1.mat',mdict={'biases_b1':biases_b1})
#sc.savemat('biases_b2.mat',mdict={'biases_b2':biases_b2})
#sc.savemat('biases_out.mat',mdict={'biases_out':biases_out})


    # Test model
    #pred = multilayer_perceptron(x, weights, biases,False)

    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({x: data[5][b'data'], y:sendvec(data[5][b'labels']),keep_prob:1}))
