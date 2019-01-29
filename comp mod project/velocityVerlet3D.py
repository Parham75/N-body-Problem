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

def exract_parameters(file_parameters, molecule):
    """
    Method to get the parameteres from a the opened file on the molecule O2 or N2
    first line contains parameters for O2 and second line for N2
    
    :param file_parameters: should be already open in main that contains the parameters
    :param molecule: String of the label of the Particle3D.
    :return: The three values D_e, r_e and alpha on O2 or N2 (they are different)
    """

    i = 0 if (molecule == "O2") else 1 # Choosing between first and second line for O2 or N2
    line = file_parameters.readlines()
    arguments = line[i].split(" ")     # Arguments of line i
    return float(arguments[1]), float(arguments[2]), float(arguments[3])

def force_dw(particle1, particle2, D_e, r_e, alpha):
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
    force = -2*alpha*D_e*(1-math.exp(-alpha*(R - r_e)))*math.exp(-alpha*(R - r_e))/R
    return force*vector_R


def pot_energy_dw(particle1, particle2, D_e, r_e, alpha):
    """
    Method to return potential energy of a particle in
    a Morse potential which is: D_e * ((1-exp[-a*(R - re)])**2 - 1)
    where R is separation between two atoms

    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :param D_e: parameter D_e from potential
    :param r_e: parameter r_e from potential
    :param alpha: parameter alpha from potential
    :return: potential energy of particle as float
    """

    vector_R = Particle3D.Vector_Separation(particle1,particle2) #separatiion vector 
    R = np.linalg.norm(vector_R)                                 #separation between two atom in the molecule
    potential = D_e * ((1-math.exp(-alpha*(R - r_e)))**2 - 1)
    return potential

def total_energy(particle1, particle2, D_e, r_e, alpha):
    """
    Method to return the total energy of the molecule
    (i.e. two particles) made up by their kinetic energies
    plus the potential energy of the bond.

    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :param D_e: parameter D_e from potential
    :param r_e: parameter r_e from potential
    :param alpha: parameter alpha from potential
    :return: total energy of molecule 
    """

    return particle1.kinetic_energy() + particle2.kinetic_energy() + pot_energy_dw(particle1, particle2, D_e, r_e, alpha)

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

# Begin main code
def main(argv1, argv2, argv3):

    # Read name of output file from command line
    if len(sys.argv)!=4:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file> <molecule name> <True or False for spin>")
        quit()
    else:
        outfile_name = sys.argv[1]
        molecule = sys.argv[2]
        spin = sys.argv[3]

    # knoowing input initialise filename based on files in directory
    if spin == "True":
        input_file_name = molecule + "withspin.txt"
        print("When we start, the molecule has a spin in the y direction.")
    else:
        input_file_name = molecule + ".txt"
        print("When we start, the molecule has no spin.")
        
    # Open input and output file
    outfile = open(outfile_name, "w")
    file_parameters = open("parameters.txt", "r")
    input_file = open(input_file_name, "r")

    # Set up simulation parameters
    D_e, r_e, alpha = exract_parameters(file_parameters, molecule)
    dt = 0.1
    numstep = 200
    time = 0.0

    # Set up two particles initial conditions and energy from input_file:

    p1 = Particle3D.from_file(input_file)
    p2 = Particle3D.from_file(input_file)
    energy = total_energy(p1, p2, D_e, r_e, alpha) # Initial energy value is the mean energy

    outfile.write("{0:f} {1:f} {2:12f}\n".format(time,np.linalg.norm(Particle3D.Vector_Separation(p1, p2)),energy))

    # Get initial force
    force_1 = force_dw(p1, p2, D_e, r_e, alpha)
    force_2 = -force_1 # Force vector 2 is just the opposite of  force vector 1

    # Initialise data lists for plotting later
    time_list = [time]
    pos_list = [p1.position]
    energy_list = [energy]

    # Start the time integration loop

    for i in range(numstep):
        # Update particle position
        p1.leap_pos2nd(dt, force_1)
        p2.leap_pos2nd(dt, force_2)

        # Update force
        force_new_1 = force_dw(p1, p2, D_e, r_e, alpha)
        force_new_2 = -force_new_1

        # Update particle velocity by averaging
        # current and new forces
        p1.leap_velocity(dt, 0.5*(force_1+force_new_1))
        p2.leap_velocity(dt, 0.5*(force_2+force_new_2))

        # Re-define force value
        force_1 = force_new_1
        force_2 = force_new_2

        # Increase time
        time = time + dt
        
        # Find new energy
        energy = total_energy(p1, p2, D_e, r_e, alpha)

        # Append information to data lists
        time_list.append(time)
        pos_list.append(np.linalg.norm(Particle3D.Vector_Separation(p1,p2)))
        energy_list.append(energy)

        # write in the output file
        outfile.write("{0:f} {1:f} {2:12f}\n".format(time,np.linalg.norm(Particle3D.Vector_Separation(p1,p2)),energy))

    # Post-simulation:
    
    # Close all files
    outfile.close()
    input_file.close()
    file_parameters.close()

   # Make vector separation and energy plots
    make_pyplot(time_list, pos_list, "vector separation")
    make_pyplot(time_list, energy_list, "Potential Energy")

    # fluctuation for energy
    mean_en = energy_list[0]
    max_en = np.max(energy_list)
    print("The energy fluctuation is {0:6f}".format(math.abs((max_en-mean_en)/mean_en)))


# Execute main method:
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])