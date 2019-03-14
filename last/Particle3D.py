"""
By : Mohammadreza Aboutalebi and Austin Morris
UUN : s1664598 and s1728541
date : March 2019

Astronomical N-body Simulation: list of Particle3D objects to hold n-bodies.

Read in planets,moon,comet and their initial positions and velocities from an input file.

"""

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
    * Two static methods to input initial data and find the separation vector
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
        self.position = pos
        self.velocity = vel
        self.mass = mass

        
    

    def __str__(self):
        """
        Define output format.
        For particle p=(label 2.0 0.5 1.0 0.5 0.5 0.5 1) this will print as
        "label x = 2.0, x = 0.5, z = 1.0"
        rest of variables in p are velocity array and mass
        """

        return self.label + " " + str(self.position[0]) + " " + str(self.position[1]) + " " + str(self.position[2])

    def mass(self):


        return self.mass


    def kinetic_energy(self):
        """
        Return kinetic energy as
        1/2*mass*vel^2
        """
        
        return 0.5*self.mass*np.linalg.norm(self.velocity)**2


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

        self.position = self.position + dt*self.velocity + 0.5*dt**2*acceleration

    @staticmethod
    def cm_velocity(Plist):

        no_parts = len(Plist)
        cm_velocity = np.zeros(3)
        mass_list = [a.mass for a in Plist]
        velocity_list = [b.velocity for b in Plist]
        momentum = np.zeros((no_parts, 3))
        for i in range(no_parts):

            momentum[i] = velocity_list[i]*mass_list[i]

        cm_velocity = np.sum(momentum, axis=0)/np.sum(mass_list)

        return cm_velocity

    
    @staticmethod
    def extract_data(in_file):
        """
        A static method to create a particle from a file. The method reads through the lines of the file.
        The form of the file lines should be:
        "label,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,mass"

        :param: A file that python can read through
        :return: particle 3D object
        """
        Plist = []
        line = in_file.readline()

        while line:
            args = line.split(",")
            label = str(args[0])
            position = np.array(args[1:4],float)
            velocity = np.array(args[4:7],float)

            mass = float(args[7])
            if  len(line)==0:
                break

            particle = Particle3D(label,position,velocity,mass)
            if  particle.position is None:
                continue
            Plist.append(particle)

            line = in_file.readline()

        return Plist

    @staticmethod
    def Vector_Separation(p1, p2):
        """
        A static method to return the vector separation between p1 and p2.

        :param p1: p1 particle postion as numpy array
        :param p2: p2 particle position as numpy array
        :return: Vector Separation
        """

        return p1.position - p2.position