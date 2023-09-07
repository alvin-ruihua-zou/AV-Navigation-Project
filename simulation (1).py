import control
import matplotlib.pyplot as plt
import math

# Kinematic Bicycle Model (total velocity)
delta = 0 # Front wheel steering angle
theta = 0 # Orientation
v = 0 # Velocity on center of mass
b = 0 # Wheel base length
a = 0 # Distance from rear wheel to COM
alpha = math.atan2(a*math.tan(delta),b) # Angle between applied velocity and vehicle frame

'''
Kinematic Bicycle Model (x velocity)

Assumptions:
1. No sliding
2. velocity is constant
'''
delta = 0 # Front wheel steering angle
theta = 0 # Orientation
v = 0 # Velocity on center of mass
b = 0 # Wheel base length
'''
dxdt = v*cos(theta)
dydt = v*sin(theta)
dthetadt = v/b*tan(delta)
'''

class kinematicBicycle:
    def __init__ (self, base, max_steer, v, dt = 0.01):
      self.base = base
      self.max_steer = max_steer
      self.v = v
      self.dt = dt
      
    
    def update(self, x, y, theta, steer_angle, dir = 1):
      new_x = x + dir*self.v*math.cos(steer_angle)*self.dt
      new_y = y + dir*self.v*math.sin(steer_angle)*self.dt
      steer_angle_adj = (-self.max_steer if steer_angle < -self.max_steer 
                    else self.max_steer if steer_angle > self.max_steer else steer_angle)
      new_theta = (theta + self.v/self.base*math.tan(steer_angle_adj))%(2*math.pi)
      return new_x,new_y,new_theta
    
    def turn(self, x, y, theta, steer_angles, t):
       for i in range(t):
          x,y,theta = self.update(x,y,theta,steer_angles[i])
       return x,y,theta

    # Turn right for half a second
    def turn_right(self, x, y, theta):
       for i in range(int(0.5/self.dt)):
          x,y,theta = self.update(x,y,theta,-math.pi/6)
       return x,y,theta

    # Turn left for half a second
    def turn_left(self, x, y, theta):
       for i in range(int(0.5/self.dt)):
          x,y,theta = self.update(x,y,theta,math.pi/6)
       return x,y,theta
    
    # Move forward for half a second
    def move_forwards(self, x, y, theta):
       for i in range(int(0.5/self.dt)):
          x,y,theta = self.update(x,y,theta,0)
       return x,y,theta
    
    # Move forward for half a second
    def move_backwards(self, x, y, theta):
       for i in range(int(0.5/self.dt)):
          x,y,theta = self.update(x,y,theta,0,-1)
       return x,y,theta

if __name__ == '__main__':
  delta = 0 # Front wheel steering angle
  theta = 0 # Orientation
  v = 2 # Velocity on center of mass
  b = 3 # Wheel base length
  model = kinematicBicycle(b, math.pi/2, v)
  x = 0
  y = 0
  theta = 0
  states = [[x,y,theta]]
  for i in range(5):
     x,y,theta = model.move_forwards(x,y,theta)
     states.append([x,y,theta])
     print(x,y,theta)
     x,y,theta = model.turn_right(x,y,theta)
     states.append([x,y,theta])
     print(x,y,theta)
  
  xx = [s[0] for s in states]
  xy = [s[1] for s in states]
  plt.plot(xx,xy)
  plt.gca().set_aspect('equal')
  plt.show()
  plt.savefig('test.png')
  print("done")