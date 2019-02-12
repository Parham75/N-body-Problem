from typing import Any

"""
By : Mohammadreza Aboutalebi
UUN : s1664598
date : Nov 2018

CMod Ex3: velocity Verlet time integration of
a molecule moving in a Morse potential.

Produces plots of the position of the particle
and its energy, both as function of time. Also
saves both to file.

potential D_e * ((1-exp[-a*(R - re)])**2 - 1)
force -2aD_e(1-exp[-a*(R - re)])exp[-a*(R - re)]vecector_R/R
D_e and a are parameters defined in the main() method
and passed to the functions that
calculate force and potential energy.

"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D


def force_dw(particle1, particle2):
    """
    Method to return the force on a particle in a Morse
    potential given by: -2aD_e(1-exp[-a*(R - r_e)])exp[-a*(R - re)]vector_R/R
    where R is separation between two atoms and vector_R is a vector from p1 to p2

    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :param D_e: parameter D_e from potential
    :param r_e: parameter r_e from potential
    :param alpha: parameter alpha from potential
    :return: force acting on particle as a Numpy array
    """

    vector_R = Particle3D.Vector_Separation(particle1,particle2)
    R = np.linalg.norm(vector_R)
    m1m2 = Particle3D.mass(particle1)*Particle3D.mass(particle1)
    force = ((1.48818E-34)*m1m2)/r**3
    return force*vector_R

def cm_velocity(p1,p2,p3):

    cm_momentum = 0
    total_mass = 0

    cm_momentum = p1.velocity*p1.mass + p2.velocity*p2.mass + p3.velocity*p3.mass
    total_mass = p1.mass + p2.mass + p3.mass

    return cm_momentum / total_mass
'''
def Vector_Separation(p1, p2):
        """
        A static method to :
        Return the vector separatin directed getiing p1 and p2

        :param p1: pi particle postion as numpy array
        :param p2: p2 particle position as numpy array
        :return: Vector Separation
        """
    return p1.position - p2.position

'''    
'''
def make_pyplot(x, y, label_y):
    """
    Method to make plots of two lists of numbers
    where the first is the time.

    :param x: x values of the plot
    :param y: y values of the plot
    :param label_y: label of the axis y
    """

    pyplot.title('velocityVerlet: '+ label_y+' against time')
    pyplot.xlabel('time')
    pyplot.ylabel(label_y)
    pyplot.plot(x, y)
    pyplot.show()
'''

# Begin main code
def main(argv1, argv2, argv3):

    # Read name of output file from command line
    if len(sys.argv)!=4:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + "<Particle input>" + "<Param input>" + "<output file>")
        quit()
    else:
        outfile_name = sys.argv[3]
        param_info = sys.argv[2]
        input_file_name = sys.argv[1]

    # Open input and output file
    outfile = open(outfile_name, "w")

    sim_param = open(param_info, "r").readlines()

    input_file = open(input_file_name, "r")

    # Set up simulation parameters
    dt = sim_param[0]
    numstep = sim_param[1]
    time = sim_param[2]

    # Set up two particles initial conditions and energy from input_file:

    Plist = Particle3D.extract_data(input_file)
    #p2 = Particle3D.extract_data(input_file)
    #p3 = Particle3D.extract_data(input_file)
    counter = 0
    outfile.write(counter)
    for p in Plist:
    	outfile.write(str(p)+"\n")
	counter+=1
    
    print(pos_list)

    
    # Get initial force
    acceleration_1 = (force_dw(p1, p2) + force_dw(p1, p3))/mass(Plist[0])
    acceleration_2 = (force_dw(p2, p3) + force_dw(p2, p1))/mass(Plist[1])
    acceleration_3 = (force_dw(p3, p1) + force_dw(p3, p2))/mass(Plist[2])

    # Initialise data lists for plotting later
    time_list = [time]
    #pos_list = [p1.position, p2.position, p3.position]

    # Start the time integration loop

    for i in range(numstep):
        # Update particle position
        Plist[0].position = p1.leap_pos2nd(dt, acceleration_1)
        Plist[1].position = p2.leap_pos2nd(dt, acceleration_2)
        Plist[2].position = p3.leap_pos2nd(dt, acceleration_3)

        # Update force
        acceleration_new_1 = (force_dw(p1, p2) + force_dw(p1, p3))/mass(p1)
        acceleration_new_2 = (force_dw(p2, p3) + force_dw(p2, p1))/mass(p2)
        acceleration_new_3 = (force_dw(p3, p1) + force_dw(p3, p2))/mass(p3)

        # Update particle velocity by averaging
        # current and new forces
        p1.leap_velocity(dt, 0.5*(acceleration_1+acceleration_new_1)) - p1.cm_velocity
        p2.leap_velocity(dt, 0.5*(acceleration_2+acceleration_new_2)) - p2.cm_velocity
        p3.leap_velocity(dt, 0.5*(acceleration_3+acceleration_new_3)) - p3.cm_velocity

        # Re-define force value
        acceleration_1 = acceleration_new_1
        acceleration_2 = acceleration_new_2
        acceleration_3 = acceleration_new_3


        # Increase time
        time = time + dt
        

        # Append information to data lists
        time_list.append(time)
        pos_list.append(p1.position, p2.position, p3.position)

        # write in the output file
        outfile.write("{0:f} {1:f} {2:12f}\n".format(time,pos_list))

    # Post-simulation:
    
    # Close all files
    outfile.close()
    input_file.close()
'''
   # Make vector separation and energy plots
    make_pyplot(time_list, pos_list, "vector separation")
    make_pyplot(time_list, energy_list, "Potential Energy")

    # fluctuation for energy
    mean_en = energy_list[0]
    max_en = np.max(energy_list)
    print("The energy fluctuation is {0:6f}".format(math.abs((max_en-mean_en)/mean_en)))
'''

# Execute main method:
if __name__ == "__main__":

    main(sys.argv[1],sys.argv[2],sys.argv[3])