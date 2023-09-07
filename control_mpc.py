import numpy as np
import control as ct
import control.optimal as opt
import control.flatsys as flat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.optimize import NonlinearConstraint
import time

l_global = 3
w_global = 1

def vehicle_update(t, x, u, params):
    # Get the parameters for the model
    l = params.get('wheelbase', 3.)         # vehicle wheelbase
    phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

    # Saturate the steering input
    phi = np.clip(u[1], -phimax, phimax)

    # Return the derivative of the state
    return np.array([
        np.cos(x[2]) * u[0],            # xdot = cos(theta) v
        np.sin(x[2]) * u[0],            # ydot = sin(theta) v
        (u[0] / l) * np.tan(phi)        # thdot = v/l tan(phi)
    ])

def vehicle_output(t, x, u, params):
    return x                            # return x, y, theta (full state)

def read_traj(filename):
  traj = list()
  f = open(filename,'r')
  for line in f:
    traj.append([float(n) for n in line.split()])
  f.close()
  return np.array(traj)  

def obs_avoidance(x,obs):
  '''
  x_len = obs[1]-obs[0]
  y_len = obs[3]-obs[2]
  r = np.sqrt((max(x_len,y_len)/4)**2+(min(x_len,y_len)/2)**2)
  if x_len > y_len:
    x_offset = x_len/4
    y_offset = 0
    c_x = obs[0]+x_offset
    c_y = obs[2]+y_len/2
  else:
    x_offset = 0
    y_offset = y_len/4
    c_x = obs[0]+x_len/2
    c_y = obs[2]+y_offset
  dist = list()
  for i in range(3):
    dist.append((x[0]-(c_x+i*x_offset))**2+(x[1]-(c_y+i*y_offset))**2-r**2)
  return dist[0] >= 0 and dist[1] >= 0 and dist[2] >= 0
  '''
  l = l_global
  w = w_global
  theta = np.arctan(w/l)
  r = np.sqrt((l/2)**2+(w/2)**2)
  points = [[x[0]+l/2*np.cos(x[2]-theta),x[1]+l/2*np.sin(x[2]-theta)],
            [x[0]-l/2*np.cos(x[2]-theta),x[1]-l/2*np.sin(x[2]-theta)],
            [x[0]+l/2*np.cos(x[2]+theta),x[1]+l/2*np.sin(x[2]+theta)],
            [x[0]-l/2*np.cos(x[2]+theta),x[1]-l/2*np.sin(x[2]+theta)]]
  '''
  [x[0]+l/2*np.cos(x[2]),x[1]+l/2*np.sin(x[2])],
  [x[0]-l/2*np.cos(x[2]),x[1]-l/2*np.sin(x[2])],
  [x[0]+w/2*np.cos(np.pi/2-x[2]),x[1]-w/2*np.sin(np.pi/2-x[2])],
  [x[0]-w/2*np.cos(np.pi/2-x[2]),x[1]+w/2*np.sin(np.pi/2-x[2])]
  '''
  min_dist = 10000
  for p in points:
    for x in range(obs[0],obs[1],5):
      for y in range(obs[2],obs[3],5):
        dist = np.sqrt((p[0]-x)**2+(p[1]-y)**2)
        if dist < min_dist:
          min_dist = dist
          if dist < 1: return -1
  return min_dist

def get_constraints(obs_list):
  obs_con = list()
  for obs in obs_list:
    obs_con.append((NonlinearConstraint,lambda x,u: obs_avoidance(x,u,obs),0,np.inf))
  return obs_con  

def mpc_solver(traj,obs,time_horizon):
  traj = np.array([traj[i] for i in range(0,traj.shape[0],10)])

  Tf = int(traj.shape[0])
  u0 = np.array([0., 0.])
  uf = np.array([0., 0.])
  x0 = traj[0]
  xf = traj[-1]
  Q = np.diag([0, 0, 0.1])          # don't turn too sharply
  R = np.diag([1, 1])               # keep inputs small
  P = np.diag([10000, 10000, 10000])   # get close to final point
  
  obs = [40,60,0,30]
  obs_fun = lambda x,u: obs_avoidance(x,obs)
  obs_con = (NonlinearConstraint,obs_fun,0,np.inf)

  state_constraint = opt.state_range_constraint(vehicle, [0, 0,-np.pi*2], [120,100,np.pi*2])
  constraints = [ opt.input_range_constraint(vehicle, [-12, -np.pi/12], [12, np.pi/12]),state_constraint, obs_con]

  timepts = np.linspace(0, time_horizon, time_horizon, endpoint=True)
  input_guess = np.outer(u0, np.ones((1, timepts.size)))
  u = np.array([u0,u0])
  for i in range(traj.shape[0]-time_horizon):
    #print(np.array(u).shape,np.array(u[:,-1]).shape)
    state_guess = traj[i:i+time_horizon].transpose()
    initial_guess = (state_guess, input_guess)
    traj_cost = opt.quadratic_cost(vehicle, Q, R, x0=traj[i+time_horizon], u0=np.array(u[:,-1]))
    term_cost = opt.quadratic_cost(vehicle, P, 0, x0=traj[i+time_horizon])
    #print(f'state: {state_guess}')
    #print(f'initial: {(initial_guess)}')
    #print(traj[i+time_horizon].shape)
    result = opt.solve_ocp(
        vehicle, timepts, traj[i], traj_cost, constraints, terminal_cost=term_cost,
        initial_guess=initial_guess, basis=flat.BezierFamily(6, T=Tf),)

    # Simulate the system dynamics (open loop)
    resp = ct.input_output_response(
        vehicle, timepts, result.inputs, traj[i],
        t_eval=np.linspace(0, time_horizon, time_horizon))
    t, y, u = resp.time, resp.outputs, np.array(resp.inputs)
    #print(traj[i+1,:])
    #print(y[:,-1])
    #print(u[:,0].shape)
    traj[i+1,:] = y[:,0]
    if i == 0:
      res_u = u[:,0]
    else:
      res_u = np.vstack([res_u,u[:,0]])

  traj_cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
  term_cost = opt.quadratic_cost(vehicle, P, 0, x0=xf)
  terminal = [ opt.input_range_constraint(vehicle, uf, uf) ]
  state_guess = traj[traj.shape[0]-time_horizon:traj.shape[0]].transpose()
  initial_guess = (state_guess, input_guess)
  result = opt.solve_ocp(
        vehicle, timepts, traj[traj.shape[0]-time_horizon], traj_cost, constraints, terminal_constraints = terminal,
        terminal_cost=term_cost, initial_guess=initial_guess, basis=flat.BezierFamily(6, T=Tf),)

  # Simulate the system dynamics (open loop)
  resp = ct.input_output_response(
      vehicle, timepts, result.inputs, traj[traj.shape[0]-time_horizon],
      t_eval=np.linspace(0, time_horizon, time_horizon))
  t, y, u = resp.time, resp.outputs, resp.inputs

  print(np.array(t).shape,np.array(y).shape,np.array(u).shape)
  print(np.array(res_u).shape,np.array(traj).shape)
  traj[traj.shape[0]-time_horizon:traj.shape[0],:] = np.array(y).transpose()
  res_u = np.hstack([res_u.transpose(), np.array(u)])
  return np.linspace(0, Tf, int(traj.shape[0]), endpoint=True),traj,res_u



# Define the vehicle steering dynamics as an input/output system
vehicle = ct.NonlinearIOSystem(
    vehicle_update, vehicle_output, states=3, name='vehicle',
    inputs=('v', 'phi'), outputs=('x', 'y', 'theta'))

#x0 = np.array([0., -2., 0.]); u0 = np.array([10., 0.])
#xf = np.array([100., 2., 0.]); uf = np.array([10., 0.])
#Tf = 10

traj = read_traj('traj.txt')

u0 = np.array([0., 0.])
uf = np.array([0., 0.])
#np.array([6,1,np.pi/6])#traj[8]

traj = np.array([traj[i] for i in range(0,traj.shape[0],5)])

Tf = int(traj.shape[0]/2/2)
x0 = traj[0]
xf = traj[-1]
print(x0,xf,traj.shape)
Q = np.diag([0, 0, 0.5])          # don't turn too sharply
R = np.diag([1, 1])               # keep inputs small
P = np.diag([10000, 10000, 10000])   # get close to final point
traj_cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
term_cost = opt.quadratic_cost(vehicle, P, 0, x0=xf)
terminal = [ opt.input_range_constraint(vehicle, uf, uf) ]
obs_fun = lambda x,u: (x[0]-50*0.1)**2+(x[1]-15*0.1)**2-(15*0.1)**2
obs = [40,60,0,30]
#obs_fun = lambda x,u: obs_avoidance(x,obs)
obs_con = (NonlinearConstraint,obs_fun,0,np.inf)
#obs_con = get_constraints([[25,75,0,50]])
state_constraint = opt.state_range_constraint(vehicle, [0, 0,-np.pi*2], [120,100,np.pi*2])
constraints = [ opt.input_range_constraint(vehicle, [0, -np.pi/10], [5, np.pi/10]),state_constraint, obs_con]
timepts = np.linspace(0, Tf, int(traj.shape[0]), endpoint=True)

input_guess = np.hstack([np.vstack([np.linspace(0,5,(round(timepts.size/2)+1)),np.zeros((round(timepts.size/2)+1))]),np.vstack([np.linspace(5,0,(round(timepts.size/2)+1)),np.zeros((round(timepts.size/2)+1))])])[:,:timepts.size]#np.outer(u0, np.ones((1, timepts.size)))
state_guess = traj.transpose()
initial_guess = (state_guess, input_guess)

s_t = time.time()
#terminal_constraints = terminal,
result = opt.solve_ocp(
    vehicle, timepts, x0, traj_cost, constraints, terminal_constraints = terminal,
    terminal_cost=term_cost, initial_guess=initial_guess, basis=flat.BezierFamily(6, T=Tf),)
e_t = time.time()
print(f'Time: {e_t-s_t}')
# Simulate the system dynamics (open loop)
resp = ct.input_output_response(
    vehicle, timepts, result.inputs, x0,
    t_eval=np.linspace(0, Tf, 100))
t, y, u = resp.time, resp.outputs, resp.inputs
print(y[:,-1])

'''
obs = [40,60,0,30]
x0 = traj[0]
xf = traj[-1]
t,y,u = mpc_solver(traj,obs,10)
print(y[:,-1])
plt.subplot(3, 1, 1)
print(np.array(t).shape,np.array(y).shape,np.array(u).shape)
resp = ct.input_output_response(
      vehicle, t, u, x0)
t, y, u = resp.time, resp.outputs, resp.inputs
print(u)
'''

#plt.imshow(env_map,cmap='gray', vmin=0, vmax=255)
plt.subplot(3, 1, 1)
plt.plot(y[0], y[1])
plt.plot(x0[0], x0[1], 'ro', xf[0], xf[1], 'ro')

#plt.gca().set_aspect('equal')
plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.subplot(3, 1, 2)
plt.plot(t, u[0])
#plt.axis([0, 10, 9.9, 10.1])
plt.xlabel("t [sec]")
plt.ylabel("u1 [m/s]")

plt.subplot(3, 1, 3)
plt.plot(t, u[1])
#plt.axis([0, 10, -0.015, 0.015])
plt.xlabel("t [sec]")
plt.ylabel("u2 [rad/s]")

plt.suptitle("Trajectory following")
plt.tight_layout()
plt.savefig('control2.png')


def update(num,traj,t):

  line = Line2D(t[:num], traj[:num])
  line.set_data(t[:num], traj[:num])
  return line,

print(y[0].shape,np.transpose(y)[:-1, 0].shape)
fig = plt.figure()
ax = plt.axes(xlim=(np.amin(y[0]), np.amax(y[0])), ylim=(np.amin(y[1]), np.amax(y[1])))

line, = ax.plot([], [], lw=2)
y = np.transpose(y)
def animate(n):
    line.set_xdata(y[:n, 0])
    line.set_ydata(y[:n, 1])
    return line,

anim = animation.FuncAnimation(fig, animate, frames=y.shape[0], interval=200)
anim.save('animation.gif')

#ani = animation.FuncAnimation(
#    fig, update, y.shape[1], fargs=(np.transpose(y), t), interval=100)
#ani.save('animation.gif', writer='Pillow', fps=2)
