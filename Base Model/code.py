import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import pickle
import imageio

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, Collo, WALL, INLET, OUTLET, CFD, layers, lb, ub):
        
        # Track errors
        self.lbfgsb_total_vector = np.array([])
        self.lbfgsb_PDE_vector = np.array([])
        self.lbfgsb_IC_vector = np.array([])
        self.lbfgsb_WALL_vector = np.array([])
        self.lbfgsb_INLET_vector = np.array([])
        self.lbfgsb_OUTLET_vector = np.array([])
        self.lbfgsb_CFD_vector = np.array([])
        
        self.epoch = 0

        self.lb = lb
        self.ub = ub
                
         # Mat. properties
        self.rho = 997
        self.mu = 0.001002
        self.U_inf = 0.025
        self.L_norm = 0.025
        self.St = 1        
        self.Re = (self.rho*self.U_inf*self.L_norm) / self.mu
        
        # Lambda
        self.lambdaPDE = 1
        self.lambdaICBC = 1
        self.lambdaCFD = 1

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]
        self.p_OUTLET = OUTLET[:, 2:3]
        
        self.x_CFD = CFD[:, 0:1]
        self.y_CFD = CFD[:, 1:2]
        self.u_CFD = CFD[:, 2:3]
        self.v_CFD = CFD[:, 3:4]   
        self.p_CFD = CFD[:, 4:5] 

        # Define layers
        self.layers = layers  

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])

        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])

        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])
        self.p_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.p_OUTLET.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        
        self.x_CFD_tf = tf.placeholder(tf.float32, shape=[None, self.x_CFD.shape[1]])
        self.y_CFD_tf = tf.placeholder(tf.float32, shape=[None, self.y_CFD.shape[1]])
        self.u_CFD_tf = tf.placeholder(tf.float32, shape=[None, self.u_CFD.shape[1]])
        self.v_CFD_tf = tf.placeholder(tf.float32, shape=[None, self.v_CFD.shape[1]])
        self.p_CFD_tf = tf.placeholder(tf.float32, shape=[None, self.p_CFD.shape[1]])

        self.u_pred, self.v_pred, self.p_pred = self.net_NS(self.x_tf, self.y_tf)[0:3]
        self.f_u_total, self.f_v_total, self.f_c_total = self.net_NS(self.x_c_tf, self.y_c_tf)[9:12]
        self.u_WALL_pred, self.v_WALL_pred = self.net_NS(self.x_WALL_tf, self.y_WALL_tf)[0:2]
        self.u_INLET_pred, self.v_INLET_pred = self.net_NS(self.x_INLET_tf, self.y_INLET_tf)[0:2]
        _, _, self.p_OUTLET_pred, self.u_x_OUTLET_pred, self.u_y_OUTLET_pred, self.v_x_OUTLET_pred, _, _, _ = self.net_NS(self.x_OUTLET_tf, self.y_OUTLET_tf)[0:9]
        self.u_CFD_pred, self.v_CFD_pred, self.p_CFD_pred = self.net_NS(self.x_CFD_tf, self.y_CFD_tf)[0:3]

        #minimizing physics residuals
        self.loss_f = tf.reduce_sum(tf.square(self.f_u_total)) \
                      + tf.reduce_sum(tf.square(self.f_v_total)) \
                      + tf.reduce_sum(tf.square(self.f_c_total))
                       
        self.loss_WALL = tf.reduce_sum(tf.square(self.u_WALL_pred)) \
                         + tf.reduce_sum(tf.square(self.v_WALL_pred)) 
                         
        self.loss_INLET = tf.reduce_sum(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                         + tf.reduce_sum(tf.square(self.v_INLET_pred-self.v_INLET_tf)) 
                         
        self.loss_OUTLET = tf.reduce_sum(tf.square((2/self.Re)*(self.u_x_OUTLET_pred)-self.p_OUTLET_pred))
                         
        self.loss_CFD = tf.reduce_sum(tf.square(self.u_CFD_pred-self.u_CFD_tf)) \
                         + tf.reduce_sum(tf.square(self.v_CFD_pred-self.v_CFD_tf)) \
                         + tf.reduce_sum(tf.square(self.p_CFD_pred-self.p_CFD_tf))
                         
        self.loss = self.lambdaPDE*(self.loss_f) + self.lambdaICBC*(self.loss_WALL + self.loss_INLET + self.loss_OUTLET) + self.lambdaCFD*(self.loss_CFD)
 
        # used to track loss
        self.PDE_total = tf.square(self.f_u_total) + \
                   tf.square(self.f_v_total) + \
                   tf.square(self.f_c_total)
        
        # self.ICBC_total = tf.square(self.u_IC_pred-self.u_IC_tf) + \
        #            tf.square(self.v_IC_pred-self.v_IC_tf) + \
        #            tf.square(self.p_IC_pred) + \
        #            tf.square(self.u_WALL_pred-self.u_WALL_tf) + \
        #            tf.square(self.v_WALL_pred-self.v_WALL_tf) + \
        #            tf.square(self.u_INLET_pred-self.u_INLET_tf) + \
        #            tf.square(self.v_INLET_pred-self.v_INLET_tf)  + \
        #            tf.square(self.p_OUTLET_pred-self.p_OUTLET_tf) 

        self.WALL_total = tf.square(self.u_WALL_pred) + \
                   tf.square(self.v_WALL_pred)   
                   
        self.INLET_total =  tf.square(self.u_INLET_pred-self.u_INLET_tf) + \
                   tf.square(self.v_INLET_pred-self.v_INLET_tf) 

        self.OUTLET_total =  tf.square((2/self.Re)*(self.u_x_OUTLET_pred)-self.p_OUTLET_pred)
                             
        self.CFD_total = tf.square(self.u_CFD_pred-self.u_CFD_tf) + \
                         tf.square(self.v_CFD_pred-self.v_CFD_tf) + \
                         tf.square(self.p_CFD_pred-self.p_CFD_tf)
                   
        self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                     var_list=self.weights + self.biases,
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                            'maxfun': 10000,
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 1.0 * np.finfo(float).eps})   
        # self.train_op = self.optimizer.minimize(self.loss)   
 
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = 0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list=self.weights + self.biases)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases  

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def save_NN(self, fileDir):

        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Save NN parameters successfully...")

    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        
        # H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalize x and y
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, x, y):

        u_v_p = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
        u = u_v_p[:,0:1]
        v = u_v_p[:,1:2]
        p = u_v_p[:,2:3]
        
        # R = 0.5
        # xEnd = 50
        # xStart = 0
        # dP = 1.5 #equivalent inlet pressure; calculated from dU = 1
        
        # u = u*(R**2-y**2)
        # v = v*(R**2-y**2)
        # p = dP*(xEnd-x)/(xEnd-xStart) + (x-xStart)*(xEnd-x)*p
        
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        # p_xx = tf.gradients(p_x, x)[0]
        # p_yy = tf.gradients(p_y, y)[0]

        f_u = u*u_x + v*u_y + p_x - (1/self.Re)*(u_xx + u_yy)
        f_v = u*v_x + v*v_y + p_y - (1/self.Re)*(v_xx + v_yy)
        f_c = u_x + v_y #continuity equation    
        
        return u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, f_u, f_v, f_c
    
    def callback(self, loss, PDE_vector, WALL_vector, INLET_vector, OUTLET_vector, CFD_vector):
        self.lbfgsb_total_vector = np.append(self.lbfgsb_total_vector, loss)
        self.lbfgsb_PDE_vector = np.append(self.lbfgsb_PDE_vector, PDE_vector)
        self.lbfgsb_WALL_vector = np.append(self.lbfgsb_WALL_vector, WALL_vector)
        self.lbfgsb_INLET_vector = np.append(self.lbfgsb_INLET_vector, INLET_vector)
        self.lbfgsb_OUTLET_vector = np.append(self.lbfgsb_OUTLET_vector, OUTLET_vector)
        self.lbfgsb_CFD_vector = np.append(self.lbfgsb_CFD_vector, CFD_vector)
        
        if self.epoch % 5 == 0:
            print('BFGS - Iter: %d, Loss: %.3e' % (self.epoch, loss))
        self.epoch = self.epoch + 1
      
    def train(self, nIter): 

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.p_OUTLET_tf: self.p_OUTLET,
                   self.x_CFD_tf: self.x_CFD, self.y_CFD_tf: self.y_CFD, self.u_CFD_tf: self.u_CFD, self.v_CFD_tf: self.v_CFD, self.p_CFD_tf: self.p_CFD}
        
        loss = np.array([])
        loss_f = np.array([])
        loss_WALL = np.array([])
        loss_INLET = np.array([])
        loss_OUTLET = np.array([])
        loss_CFD = np.array([])
        
        start_time = time.time()
        

        for it in range(nIter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)  
            loss_f_value = self.sess.run(self.loss_f, tf_dict)
            loss_WALL_value = self.sess.run(self.loss_WALL, tf_dict)
            loss_INLET_value = self.sess.run(self.loss_INLET, tf_dict)
            loss_OUTLET_value = self.sess.run(self.loss_OUTLET, tf_dict)   
            loss_CFD_value = self.sess.run(self.loss_CFD, tf_dict)    
            
            
            # Print
            if it % 5 == 0:
                elapsed = time.time() - start_time
                print('Adam - Iter: %d, Loss: %.3e, Time: %.2f' % (it, loss_value, elapsed))
                start_time = time.time()
            
            loss = np.append(loss, loss_value)
            loss_f = np.append(loss_f, loss_f_value)
            loss_WALL = np.append(loss_WALL, loss_WALL_value)
            loss_INLET = np.append(loss_INLET, loss_INLET_value)
            loss_OUTLET = np.append(loss_OUTLET, loss_OUTLET_value)
            loss_CFD = np.append(loss_CFD, loss_CFD_value)
        
        # self.optimizer_Adam.minimize(self.loss)
        
        adam_total_vector = loss
        adam_PDE_vector = loss_f      
        adam_WALL_vector = loss_WALL       
        adam_INLET_vector = loss_INLET     
        adam_OUTLET_vector = loss_OUTLET
        adam_CFD_vector = loss_CFD

        PDE_total = self.sess.run(self.PDE_total, tf_dict)
        WALL_total = self.sess.run(self.WALL_total, tf_dict)  
        INLET_total = self.sess.run(self.INLET_total, tf_dict) 
        OUTLET_total = self.sess.run(self.OUTLET_total, tf_dict)  
        CFD_total = self.sess.run(self.CFD_total, tf_dict)  
        
        self.optimizer_BFGS.minimize(self.sess,
                feed_dict = tf_dict,
                fetches = [self.loss, self.loss_f, self.loss_WALL, self.loss_INLET, self.loss_OUTLET, self.loss_CFD],
                loss_callback = self.callback)
        
        return adam_total_vector, adam_PDE_vector, adam_WALL_vector, adam_INLET_vector, adam_OUTLET_vector, adam_CFD_vector, \
            PDE_total, WALL_total, INLET_total, OUTLET_total, CFD_total,\
                self.lbfgsb_total_vector, self.lbfgsb_PDE_vector, self.lbfgsb_WALL_vector, self.lbfgsb_INLET_vector, self.lbfgsb_OUTLET_vector, self.lbfgsb_CFD_vector
        
    def predict(self, x_star, y_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
    
        return u_star, v_star, p_star
    
def CartGrid(xmin, xmax, ymin, ymax, num_x, num_y):
    # num_x, num_y: number per edge
    # num_t: number time step

    x = np.linspace(xmin, xmax, num=num_x)
    y = np.linspace(ymin, ymax, num=num_y)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy    
    
    
if __name__ == "__main__": 
      
    # Domain bounds
    xmax = 50
    xmin = 0
    ymax = 0.5
    ymin = -0.5
    lb = np.array([xmin, ymin])
    ub = np.array([xmax, ymax])
    
    # Select DNN architecture
    layers = [2] + 5*[50] + [3]

    # Create points
    x_upb, y_upb = CartGrid(xmin=xmin, xmax=xmax,
                                   ymin=ymax, ymax=ymax,
                                   num_x=251, num_y=1)

    x_lwb, y_lwb = CartGrid(xmin=xmin, xmax=xmax,
                                   ymin=ymin, ymax=ymin,
                                   num_x=251, num_y=1)
    
    wall_up = np.concatenate((x_upb, y_upb), 1)
    wall_lw = np.concatenate((x_lwb, y_lwb), 1)    
    WALL = np.concatenate((wall_up, wall_lw), 0)
    
    print('Total WALL Points: {}'.format(WALL.shape))

    x_inb, y_inb = CartGrid(xmin=xmin, xmax=xmin,
                                   ymin=ymin, ymax=ymax,
                                   num_x=1, num_y=51)
    U_max = 1.0
    # u_inb = 4*U_max*(0.25-y_inb**2)
    u_inb = np.ones_like(x_inb)
    v_inb = np.zeros_like(x_inb)
    INB = np.concatenate((x_inb, y_inb, u_inb, v_inb), 1)  
    
    print('Total INB Points: {}'.format(INB.shape))

    x_outb, y_outb = CartGrid(xmin=xmax, xmax=xmax,
                                      ymin=ymin, ymax=ymax,
                                      num_x=1, num_y=51)
    p_outb = np.zeros_like(x_outb)
    OUTB = np.concatenate((x_outb, y_outb, p_outb), 1)
    
    print('Total OUTB Points: {}'.format(OUTB.shape))

    # Load CFD Data
    data = sio.loadmat('data_CFD.mat')

    u_cfd = data['U']
    v_cfd = data['V']
    p_cfd = data['P']
    x_cfd = data['x'] 
    y_cfd = data['y'] 

    u_cfd = u_cfd[:,:,-1].T
    v_cfd = v_cfd[:,:,-1].T
    p_cfd = p_cfd[:,:,-1].T
    x_cfd = x_cfd[0] # After transpose: 51 x 1 array
    y_cfd = y_cfd[0] # After transpose: 51 x 1 array

    x_cfd, y_cfd = np.meshgrid(x_cfd, y_cfd)
    x_cfd = x_cfd.flatten()[:, None]
    y_cfd = y_cfd.flatten()[:, None]
    u_cfd = u_cfd.flatten()[:, None]
    v_cfd = v_cfd.flatten()[:, None]
    p_cfd = p_cfd.flatten()[:, None]

    XY_CFD = np.concatenate((x_cfd, y_cfd, u_cfd, v_cfd, p_cfd), 1)   
    
    sample_CFD = np.random.randint(XY_CFD.shape[0], size=100) #sample points from CFD
    XY_CFD = XY_CFD[sample_CFD,:]
    
    print('Total CFD Points: {}'.format(XY_CFD.shape))

    # Randomize collocation points throughout domain
    # x_c, y_c = CartGrid(xmin=xmin, xmax=xmax,
    #                     ymin=ymin, ymax=ymax,
    #                     num_x=451, num_y=56)
    # XY_c = np.concatenate((x_c, y_c), 1)

    sample_count = 5000
    xrange = (xmin, xmax)
    yrange = (ymin, ymax)
        
    x_c = []
    y_c = []
    [x_c.append(np.random.uniform(*xrange)) for i in range(sample_count)]
    [y_c.append(np.random.uniform(*yrange)) for i in range(sample_count)]
    
    XY_c = np.column_stack((x_c, y_c))
    
    print('Total Random Points: {}'.format(XY_c.shape))
    
    
    # Visualize ALL the training points
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1, s=2, color='blue')
    ax.scatter(WALL[:, 0:1], WALL[:, 1:2], marker='o', alpha=0.1, s=2, color='orange')
    ax.scatter(INB[:, 0:1], INB[:, 1:2], marker='o', alpha=0.1, s=2, color='yellow')
    ax.scatter(OUTB[:, 0:1], OUTB[:, 1:2], marker='o', alpha=0.1, s=2, color='green')
    ax.scatter(XY_CFD[:, 0:1], XY_CFD[:, 1:2], marker='o', alpha=0.1, s=2, color='black')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.savefig('training_points.png')
    plt.close()
    
    
    # Combine all collocation points
    XY_c = np.concatenate((XY_c, WALL, OUTB[:,0:2], INB[:,0:2], XY_CFD[:,0:2]), 0)

    print('Total Collocation Points: {}'.format(XY_c.shape))



    # Training Model
    start_time = time.time()
    model = PhysicsInformedNN(XY_c, WALL, INB, OUTB, XY_CFD, layers, lb, ub)
    adam_total_vector, adam_PDE_vector, adam_WALL_vector, adam_INLET_vector, adam_OUTLET_vector, adam_CFD_vector,  \
        PDE_total, WALL_total, INLET_total, OUTLET_total, CFD_total, \
            lbfgsb_total_vector, lbfgsb_PDE_vector, lbfgsb_WALL_vector, lbfgsb_INLET_vector, lbfgsb_OUTLET_vector, lbfgsb_CFD_vector = model.train(10000)
    
   # Save neural network
    model.save_NN('PINN.pickle')       
    
    # Error vector determination
    N_train = len(XY_c)
    adam_total_vector /= N_train
    adam_PDE_vector /= N_train
    adam_WALL_vector /= N_train
    adam_INLET_vector /= N_train
    adam_OUTLET_vector /= N_train
    adam_CFD_vector /= N_train
    lbfgsb_total_vector /= N_train
    lbfgsb_PDE_vector /= N_train
    lbfgsb_WALL_vector /= N_train
    lbfgsb_INLET_vector /= N_train
    lbfgsb_OUTLET_vector /= N_train
    lbfgsb_CFD_vector /= N_train

    # Prediction
    Nx = 750
    Ny = 50
    
    x_star = np.linspace(xmin, xmax, Nx)
    y_star = np.linspace(ymin, ymax, Ny)
    X_plot, Y_plot = np.meshgrid(x_star, y_star)
    x_test = X_plot.flatten()[:, None]
    y_test = Y_plot.flatten()[:, None]

    x_data = x_star
    y_data = y_star  
    u_data = np.zeros((Nx, Ny))
    v_data = np.zeros((Nx, Ny))
    p_data = np.zeros((Nx, Ny))
    
    u_pred, v_pred, p_pred = model.predict(x_test, y_test)

    arr_x = np.reshape(x_test, (Ny, Nx)).T
    arr_y = np.reshape(y_test, (Ny, Nx)).T        
    arr_u = np.reshape(u_pred, (Ny, Nx)).T
    arr_v = np.reshape(v_pred, (Ny, Nx)).T
    arr_p = np.reshape(p_pred, (Ny, Nx)).T
    
    u_data[:,:] = arr_u
    v_data[:,:] = arr_v
    p_data[:,:] = arr_p
    
    plt.rcParams.update({'font.size': 18})
    
    fig_2 = plt.figure(figsize=(20,20), dpi=100)

    ax = fig_2.add_subplot(311)
    plt.contourf(arr_x, arr_y, arr_u, cmap=cm.viridis)
    plt.colorbar()
    plt.title('Predicted U')  
    plt.xlabel('X')
    plt.ylabel('Y')

    ax = fig_2.add_subplot(312)
    plt.contourf(arr_x, arr_y, arr_v, cmap=cm.viridis)
    plt.colorbar()
    plt.title('Predicted V')  
    plt.xlabel('X')
    plt.ylabel('Y')

    ax = fig_2.add_subplot(313)
    plt.contourf(arr_x, arr_y, arr_p, cmap=cm.viridis)
    plt.colorbar()
    plt.title('Predicted Pressure') 
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig('U_V_P Plots.png')
    plt.close()     
     
    # Error calculation
    total_PDE_loss = np.sum(PDE_total)/(N_train)
    total_WALL_loss = np.sum(WALL_total)/(N_train)
    total_INLET_loss = np.sum(INLET_total)/(N_train)
    total_OUTLET_loss = np.sum(OUTLET_total)/(N_train)
    total_CFD_loss = np.sum(CFD_total)/(N_train)
    total_ICBC_loss = total_WALL_loss + total_INLET_loss + total_OUTLET_loss
    total_loss = total_PDE_loss + total_ICBC_loss + total_CFD_loss
    print('Total Loss: %e' % (total_loss))
    print('Total PDE Loss: %e' % (total_PDE_loss))
    print('Total ICBC Loss: %e' % (total_ICBC_loss))
    print('Total CFD Loss: %e' % (total_CFD_loss))
    print("Time Taken: %e" %(time.time() - start_time))
    
    adam_ICBC_vector = adam_total_vector - (adam_PDE_vector + adam_CFD_vector)
    lbfgsb_ICBC_vector = lbfgsb_total_vector - (lbfgsb_PDE_vector + lbfgsb_CFD_vector)
    
    plt.rcParams.update({'font.size': 10})
    
    # ADAM optimizer
    fig_err, ax_err1 = plt.subplots()
    ax_err1.set_xlabel('Epoch')
    ax_err1.set_ylabel('ADAM Loss', color = 'tab:blue')
    ax_err1.set_yscale('log')
    ax_err1.plot(adam_total_vector, color = 'tab:red', label = 'Total Loss')
    ax_err1.plot(adam_PDE_vector, color = 'tab:green', label = 'PDE Loss')
    ax_err1.plot(adam_ICBC_vector, color = 'tab:blue', label = 'ICBC Loss')
    ax_err1.plot(adam_CFD_vector, color = 'tab:purple', label = 'CFD Loss')
    ax_err1.legend(loc=0)
    fig_err.tight_layout()
    plt.savefig('error_vs_epochADAM.png')
    #plt.show()
    plt.close()
 
    
    # L-BFGS-B optimizer 
    fig_err, ax_err1 = plt.subplots()
    ax_err1.set_xlabel('Epoch')
    ax_err1.set_ylabel('LBFGSB Loss', color = 'tab:blue')
    ax_err1.set_yscale('log')
    ax_err1.plot(lbfgsb_total_vector, color = 'tab:red', label = 'Total Loss')
    ax_err1.plot(lbfgsb_PDE_vector, color = 'tab:green', label = 'PDE Loss')
    ax_err1.plot(lbfgsb_ICBC_vector, color = 'tab:blue', label = 'ICBC Loss')
    ax_err1.plot(lbfgsb_CFD_vector, color = 'tab:purple', label = 'CFD Loss')
    ax_err1.legend(loc=0)
    fig_err.tight_layout()
    plt.savefig('error_vs_epochLBFGSB.png')
    plt.close()

    # Plot errors
    plt.rcParams.update({'font.size': 20})
    
    fig_PDE = plt.figure(figsize=(20,20), dpi=100)
    
    ax = plt.axes()
    ax.scatter(XY_c[:,0], XY_c[:,1], c = PDE_total, cmap = 'Reds')
    ax.set_title('PDE Residuals')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.savefig('PDEerrors.png')
    plt.close()

    fig_ICBC = plt.figure(figsize=(20,20), dpi=100)
    
    ax = plt.axes()
    ax.set_title('ICBC Residuals')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.scatter(WALL[:,0], WALL[:,1], c = WALL_total, cmap = 'Greens', label = 'WALL')
    ax.scatter(INB[:,0], INB[:,1], c = INLET_total, cmap = 'Reds', label = 'INLET')
    ax.scatter(OUTB[:,0], OUTB[:,1], c = OUTLET_total, cmap = 'Blues', label = 'OUTLET')
    ax.legend(loc=1)
    plt.savefig('ICBCerrors.png')
    plt.close()

    fig_CFD = plt.figure(figsize=(20,20), dpi=100)

    ax = plt.axes()
    ax.set_title('CFD Residuals')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.scatter(XY_CFD[:,0], XY_CFD[:,1], c = CFD_total, cmap = 'Reds')
    ax.legend(loc=1)
    plt.savefig('CFDerrors.png')
    plt.close()

#export data to .mat
sio.savemat('data_PINN.mat', {'U': u_data, 'V': v_data, 'P': p_data, 'x': x_data, 'y': y_data})




