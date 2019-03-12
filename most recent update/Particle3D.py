"""
 CMod Ex3: Particle3D, a class to describe 3D particles
 By : Mohammadreza Aboutalebi
 UUN : s1664598
 date : Nov 2018
"""

'''
    with open(param_info, "r") as ins:
        sim_param = []
    print(type(ins))
    for line in ins:
        sim_param.append(line)
'''
import math
import numpy as np
import types

class Particle3D(object):
    """
    Class to describe 3D particles.

    Properties:
    position(float) - position vector as numpy array
    velocity(float) - velocity vector as numpy array
    mass(float) - particle mass

    Methods:
    * formatted output
    * kinetic energy
    * first-order velocity update
    * first- and second order position updates
    * Two static methods for inputing initial data and find the separation vector
    """
    def __init__(self, label, pos, vel, mass):
    #def __init__(self, label, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, mass):
        """
        Initialise a Particle3D instance
        
        :param label: label as string
        :param pos: position as float
        :param vel: velocity as float
        :param mass: mass as float
        """

        self.label = label
        #self.position = np.array([x_pos, y_pos, z_pos])
        #self.velocity = np.array([x_vel, y_vel, z_vel])
        self.position = pos
        self.velocity = vel
        self.mass = mass

        
    

    def __str__(self):
        """
        Define output format.
        For particle p=(label 2.0 0.5 1.0 0.5 0.5 0.5 1) this will print as
        "label x = 2.0, x = 0.5, z = 1.0"
        rest of variables in p are velosity array and mass
        """

        return self.label + " " + str(self.position[0]) + " " + str(self.position[1]) + " " + str(self.position[2])
    '''
    def position(self):


        return self.position 
    ''' 
  
    def mass(self):


        return self.mass

    def kinetic_energy(self):
        """
        Return kinetic energy as
        1/2*mass*vel^2
        """
        
        return 0.5*self.mass*np.linalg.norm(self.velocity)**2

    # Time integration methods

    def leap_velocity(self, dt, acceleration):
        """
        First-order velocity update,
        v(t+dt) = v(t) + dt*F(t)/m

        :param dt: timestep as float
        :param force: force on particle as float
        """
        #if type(self.velocity) != np.ndarray:

        self.velocity = self.velocity + dt*acceleration

    def leap_pos1st(self, dt):
        """
        First-order position update,
        x(t+dt) = x(t) + dt*v(t)

        :param dt: timestep as float
        """

        self.position = self.position + dt*self.velocity
        #if  self.position is None:
                #raise TypeError

    def leap_pos2nd(self, dt, acceleration):
        """
        Second-order position update,
        x(t+dt) = x(t) + dt*v(t) + 1/2*dt^2*F(t)

        :param dt: timestep as float
        :param force: current force as float
        """
        #acceleration = self.acceleration


        self.position = self.position + dt*self.velocity + 0.5*dt**2*acceleration
        #if  self.position is None:
                #raise TypeError

    @staticmethod
    def extract_data(in_file):
        """
        A static method to :
        create a particle from a file. the method read through the line of file.
        The form of the file line should be:
        "label pos_x pos_y pos_z vel_x vel_y vel_z mass"

        :param: A file that python can read through
        :return: particle 3D object
        """
        Plist = []
        line = in_file.readline()
        #print(line, "halim")
        #for line in range(len(line))
        while line:
            #print(line)
            args = line.split(",")
            label = str(args[0])
            position = np.array(args[1:4],float)
            velocity = np.array(args[4:7],float)
            #x_pos = float(args[1])
            #y_pos = float(args[2])
            #z_pos = float(args[3])
            #x_vel = float(args[4])
            #y_vel = float(args[5])
            #z_vel = float(args[6])
            mass = float(args[7])
            if  len(line)==0:
                break
            #print(args, label, position, velocity, mass)
            particle = Particle3D(label,position,velocity,mass)
            if  particle.position is None:
                continue
            Plist.append(particle)
            #Plist.append(Particle3D(label, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, mass))
            line = in_file.readline()

            #if 'str' in line:
            #if  len(line)==0:
                #break

            #print(line, "holo") 
        #print(np.vstack([o.position for o in Plist]), "holo")   
        return Plist

    #@staticmethod
    def Vector_Separation(p1, p2):
        """
        A static method to : when i change line 111 from self. acceleration to acceleration this deoes not work anymore
        Return the vector separatin directed getiing p1 and p2

        :param p1: pi particle postion as numpy array
        :param p2: p2 particle position as numpy array
        :return: Vector Separation
        """
        """
        if type(p1.position) == np.ndarray:

            return p1.position - p2.position

        else:

            return 0
        """
        #vec_sep = np.zeros(3)
        #if p1.position is not None and p2.position is not None:

            #vec_sep = p1.position - p2.position
        #vec_sep = np.zeros(3)
        #while p1.position is not None:

            #vec_sep = p1.position - p2.position
        #if isinstance(p1.position - p2.position,):
        #vec_sep = (p1.position - p2.position if p1.position is not None)

        #for x in vec_sep:
            #return x

        #if any(p1.position) is not None:
            #return p1.position - p2.position
        #print(type(p1.position))

        #print(p1.position, "havij")
        return p1.position - p2.position
        #return vec_sep


