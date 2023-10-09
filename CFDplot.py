import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#load data from .mat
data = sio.loadmat('data_CFD.mat')

u_cfd = data['U']
v_cfd = data['V']
p_cfd = data['P']
x_cfd = data['x'] 
y_cfd = data['y'] 

u_avg = u_cfd[:,:,-1].mean(axis = 1)
v_avg = v_cfd[:,:,-1].mean(axis = 1)
p_avg = p_cfd[:,:,-1].mean(axis = 1)

plt.rcParams.update({'font.size': 10})    

xshape = u_cfd.shape[0]
yshape = u_cfd.shape[1]
onex = int(xshape / 50)
midy = int(yshape / 2)

for i in range(1,10): #plot 9 total points
    loc = i*onex*5 #increments of x = 5
    curr_vel = np.sqrt(np.square(u_cfd[loc,:,-1])**2+np.square(v_cfd[loc,:,-1])**2)
    lbl = "x = " + str(i*5)
    plt.plot(curr_vel, y_cfd[0], label=lbl)
    
plt.title("Velocity Profile")
plt.xlabel("Velocity")
plt.ylabel("Length (y)")
plt.legend()
plt.savefig("VelocityProfile.png")
plt.close()

plt.plot(x_cfd[0][:9*onex*5], np.sqrt(u_cfd[:9*onex*5,25,-1]**2+v_cfd[:9*onex*5,25,-1]**2))
plt.title("Total Velocity Across Length")
plt.xlabel("Length (x)")
plt.ylabel("Velocity")
plt.savefig("TotalVelocityLength.png")
plt.close()

plt.plot(x_cfd[0][:9*onex*5], p_avg[:9*onex*5])
plt.title("Pressure Across Length")
plt.xlabel("Length (x)")
plt.ylabel("Pressure")
plt.savefig("PressureLength.png")
plt.close()

#subsample every 25 points
p_sample = p_avg[::25]
x_sample = x_cfd[0][::25]

dpdx = [0.0]*len(x_sample)
dpdx[0] = (p_sample[0] - p_sample[1])/(x_sample[0] - x_sample[1])
for i in range(1,len(p_sample)-1):
    dpdx[i] = (p_sample[i+1] - p_sample[i-1])/(x_sample[i+1]-x_sample[i-1])
dpdx[-1] = (p_sample[-1] - p_sample[-2])/(x_sample[-1] - x_sample[-2])

plt.plot(x_sample[2:-1], dpdx[2:-1])
plt.title("Pressure Gradient Across Length")
plt.xlabel("Length (x)")
plt.ylabel("dp/dx")
plt.savefig("PressureGradientLength.png")
plt.close()

dp2d2x = [0.0]*len(x_sample)
dp2d2x[0] = (dpdx[0] - dpdx[1])/(x_sample[0] - x_sample[1])
for i in range(1,len(dpdx)-1):
    dp2d2x[i] = (dpdx[i+1] - dpdx[i-1])/(x_sample[i+1]-x_sample[i-1])
dp2d2x[-1] = (dpdx[-1] - dpdx[-2])/(x_sample[-1] - x_sample[-2])

plt.plot(x_sample[2:-1], dp2d2x[2:-1])
plt.title("Pressure Curvature Across Length")
plt.xlabel("Length (x)")
plt.axhline(0, color="red", alpha=0.3)
plt.ylabel("dp^2/d^2x")
plt.savefig("PressureCurvatureLength.png")
plt.close()





