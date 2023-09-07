import numpy as np
import control as ct
import control.optimal as opt
import control.flatsys as flat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.optimize import NonlinearConstraint

class motionPrim:
    def __init__ (self, prim_id, start_angle, endpose, costmult,inter_poses, num_interposes):
        self.id = prim_id
        self.start_angle = start_angle
        self.endpose = endpose
        self.cost = costmult
        self.inter_poses = inter_poses
        self.num_interposes = num_interposes
        self.cells_covered = list()
        self.inputs = list()


class motionPrims:
    def __init__ (self,prims,num_prims,resolution,num_angles):
        self.prims = prims
        self.num_prims = num_prims
        self.resolution = resolution
        self.num_angles = num_angles


def get_prims(prims_file):
    f = open(prims_file,'r')
    resolution = float(f.readline().split(":")[1])
    num_angles = int(f.readline().split(":")[1])
    num_prims = int(f.readline().split(":")[1])
    motion_prims = list()
    for n in range(num_prims):
        prim_id = int(f.readline().split(":")[1])
        prim_id = n
        start_angle = int(f.readline().split(":")[1])
        endpose = [int(n) for n in f.readline().split(":")[1].split()]
        #multcost = 1 if int(f.readline().split(":")[1]) < 5 else 10
        multcost =int(f.readline().split(":")[1])+ (abs(endpose[0])+abs(endpose[1]))
        num_interposes = int(f.readline().split(":")[1])
        interposes = list()
        for i in range(num_interposes):
            interposes.append([float(n.strip()) for n in f.readline().split()])
        motion_prims.append(motionPrim(prim_id,start_angle,endpose,multcost,interposes,num_interposes))
    f.close()
    return motionPrims(motion_prims,num_prims,resolution,num_angles)

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

vehicle = ct.NonlinearIOSystem(
    vehicle_update, vehicle_output, states=3, name='vehicle',
    inputs=('v', 'phi'), outputs=('x', 'y', 'theta'))


prims = get_prims('prim2.txt')
for i in range(5):
  time = 0.2
  while time < 5:
    u0 = np.array([1., 0.])
    uf = np.array([1., 0.])
    traj = np.array(prims.prims[i].inter_poses)
    Tf = time
    time += 0.3
    x0 = traj[0]
    xf = traj[-1]
    Q = np.diag([0, 0, 0.1])          # don't turn too sharply
    R = np.diag([1, 1])               # keep inputs small
    P = np.diag([10000, 10000, 10000])   # get close to final point
    traj_cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
    term_cost = opt.quadratic_cost(vehicle, P, 0, x0=xf)
    #obs_con = get_constraints([[25,75,0,50]])
    state_constraint = opt.state_range_constraint(vehicle, [0, 0,-np.pi*2], [120,100,np.pi*2])
    if xf[0] + xf[1] <= 0:
      constraints = [ opt.input_range_constraint(vehicle, [-3, -np.pi/12], [5, np.pi/12]),state_constraint]   
    else:
      constraints = [ opt.input_range_constraint(vehicle, [0, -np.pi/12], [5, np.pi/12]),state_constraint]

    timepts = np.linspace(0, Tf, int(traj.shape[0]), endpoint=True)

    input_guess = np.outer(u0, np.ones((1, timepts.size)))
    state_guess = traj.transpose()
    initial_guess = (state_guess, input_guess)
    epsilon = [0.08,0.08,0.05]
    terminal = [opt.input_range_constraint(vehicle, uf-0.01, uf+0.01),
                opt.state_range_constraint(vehicle, np.array(prims.prims[i].endpose)*prims.resolution-epsilon, np.array(prims.prims[i].endpose)*prims.resolution+epsilon)]
    result = opt.solve_ocp(
        vehicle, timepts, x0, traj_cost, constraints, terminal_constraints = terminal,
        terminal_cost=term_cost, initial_guess=initial_guess, basis=flat.BezierFamily(6, T=Tf),)
    print(f'i: {i}, time: {time}, desired: {xf} state: {result.states[:,-1]} succ: {result.success}')
    if not result.success: continue
    # Simulate the system dynamics (open loop)
    resp = ct.input_output_response(
        vehicle, timepts, result.inputs, x0,
        t_eval=np.linspace(0, Tf, 10))
    t, y, u = np.array(resp.time), np.array(resp.outputs), np.array(resp.inputs)
    print(np.array(t).shape,np.array(y).shape,np.array(u).shape)
    print(y.transpose())
    prims.prims[i].inter_poses = y.transpose()
    prims.prims[i].inputs = u.transpose()
    print(f'i: {i}, time: {time}')
    break

i = 0
for prim in prims.prims:
  xx = [s[0] for s in prim.inter_poses]
  xy = [s[1] for s in prim.inter_poses]
  plt.plot(xx,xy)
  plt.gca().set_aspect('equal')
  plt.show()
  
  i += 1
  if i ==5: break
plt.savefig('prims.png')
exit()
plt.subplot(3, 1, 1)
plt.plot(y[0], y[1])
plt.plot(x0[0], x0[1], 'ro', xf[0], xf[1], 'ro')

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

plt.suptitle("Lane change manuever")
plt.tight_layout()
plt.savefig('control_prim.png')
