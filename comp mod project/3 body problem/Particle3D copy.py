"""
 CMod Ex3: Particle3D, a class to describe 3D particles
 By : Mohammadreza Aboutalebi
 UUN : s1664598
 date : Nov 2018
"""
import math
import numpy as np

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
        """
        Initialise a Particle3D instance
        
        :param label: label as string
        :param pos: position as float
        :param vel: velocity as float
        :param mass: mass as float
        """
 
        self.position = pos
        self.velocity = vel
        self.mass = mass
        self.label = label
    

    def __str__(self):
        """
        Define output format.
        For particle p=(label 2.0 0.5 1.0 0.5 0.5 0.5 1) this will print as
        "label x = 2.0, x = 0.5, z = 1.0"
        rest of variables in p are velosity array and mass
        """

        return self.label + " x = " + str(self.position[0]) + ", y = " + str(self.position[1]) + ", z = " + str(self.position[2])
    
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

        self.velocity = self.velocity + dt*acceleration

    def leap_pos1st(self, dt):
        """
        First-order position update,
        x(t+dt) = x(t) + dt*v(t)

        :param dt: timestep as float
        """

        self.position = self.position + dt*self.velocity

    def leap_pos2nd(self, dt, acceleration):
        """
        Second-order position update,
        x(t+dt) = x(t) + dt*v(t) + 1/2*dt^2*F(t)

        :param dt: timestep as float
        :param force: current force as float
        """

        self.position = self.position + dt*self.velocity + 0.5*dt**2*self.acceleration

    @staticmethod
    def extract_data(file_handle):
        """
        A static method to :
        create a particle from a file. the method read through the line of file.
        The form of the file line should be:
        "label pos_x pos_y pos_z vel_x vel_y vel_z mass"

        :param: A file that python can read through
        :return: particle 3D object
        """

        line = file_handle.readline()
        args = line.split(" ")
        label = args[0]
        position = np.array(args[1:4],float)
        velocity = np.array(args[4:7],float)
        mass = float(args[7])
        return Particle3D(label,position,velocity,mass)

    @staticmethod
    def Vector_Separation(p1, p2):
        """
        A static method to :
        Return the vector separatin directed getiing p1 and p2

        :param p1: pi particle postion as numpy array
        :param p2: p2 particle position as numpy array
        :return: Vector Separation
        """
    	return p1.position - p2.position

