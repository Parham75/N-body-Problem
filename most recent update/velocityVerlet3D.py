from typing import Any

"""
By : Mohammadreza Aboutalebi and Austin Morris
UUN : s1664598 and s1728541
date : March 2019

Astronomical N-body Simulation: velocity Verlet time integration of
objects moving in a Gravitational potential.

Correct for initial centre-of-mass motion,
Simulate the evolution of the system using the velocity Verlet time integration algorithm,
Write a trajectory file for the simulation that can be visualised using VMD.

potential -(G*m1m2)/R)
force -(G*m1m2)/R**3)*vector_R
R and and vector_R are parameters defined in the force_dw() method
and passed to the functions that
calculate force and potential energy.

"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D


def exract_parameters(file_parameters):
    """
    Method to get the parameteres from the opened file on the astronomical bodies
    
    :param file_parameters: should be already open in main that contains the parameters
    :param planet/moon/comet: String of the label of the Particle3D.
    :return: The three values dt, number of steps, and total time
    """
    line = file_parameters.readlines()

    return float(line[0]), int(line[1]), float(line[2])

def force_dw(Plist):
    """
    Method to return the force on a particle in a Gravitational
    potential given by: -(G*m1m2)/R**3)*vector_R
    where R is separation between two bodies and vector_R is a vector from pi to pj

    :param particlei: Particle3D instance
    :param particlej: Particle3D instance
    :param G: constant parameter
    :param vector_R: parameter vector_R from potential
    :param R: parameter R from potential
    :return: force acting on body as a Numpy array
    """
    no_parts = len(Plist)
    #pos_list = [o.position for o in Plist]
    force_dw = np.zeros((no_parts, no_parts, 3))
    
    for i,pi in enumerate(Plist):
        for j,pj in enumerate(Plist):
            vector_R = Particle3D.Vector_Separation(pi, pj)
            #vector_R = pos_list[i] - pos_list[j]
            R = np.linalg.norm(vector_R)
            m1m2 = Particle3D.mass(pi)*Particle3D.mass(pj)
            #m1m2 = Plist[pi].mass*Plist[pj].mass
            #if pi != pj:
            if R != 0:
                force_dw[i, j, :] = (((-1.48818E-34)*m1m2)/R**3)*vector_R
                #G is defined as the Gravitational constant and = -1.48818E-34 au^3/day^2/kg

            else:
                force_dw[i, j, :] = np.zeros(3)

    return force_dw

def acceleration(Plist, force):

    no_parts = len(Plist)
    acceleration = np.zeros((no_parts, no_parts, 3))
    for i in range(no_parts):

        acceleration[i] = np.sum(force[i]/Plist[i].mass , axis=0)

    return acceleration

def pot_energy_dw(Plist):
    """
    Method to return potential energy of a body in
    a Gravitational potential which is given by:
    -(G*m1m2)/R)

    :param particlei: Particle3D instance
    :param particlej: Particle3D instance
    :param G: constant parameter
    :param vector_R: parameter vector_R from potential
    :param R: parameter R from potential
    :return: potential energy of body as float
    """

    no_parts = len(Plist)
    #pos_list = [o.position for o in Plist]
    pot_energy_dw = np.zeros((no_parts, no_parts))
    
    for i,pi in enumerate(Plist):
        for j,pj in enumerate(Plist):
            vector_R = Particle3D.Vector_Separation(pi, pj)
            #vector_R = pos_list[i] - pos_list[j]
            R = np.linalg.norm(vector_R)
            m1m2 = Particle3D.mass(pi)*Particle3D.mass(pj)
            #m1m2 = Plist[pi].mass*Plist[pj].mass
            #if pi != pj:
            if R != 0:
                pot_energy_dw[i, j] = ((-(1.48818E-34)*m1m2)/R)

            else:
                pot_energy_dw[i, j] = 0

    return pot_energy_dw

def kinetic_energy_dw(Plist):


    no_parts = len(Plist)
    kinetic_energy = np.zeros(no_parts)
    for i in range(no_parts):

        kinetic_energy[i] = Particle3D.kinetic_energy(Plist[i])


    return kinetic_energy



def period(pos_array_in, pos_list_each, time_list):

    cosinee = np.zeros(len(pos_list_each))
    periode = np.zeros(len(time_list))
    #print(pos_list_each)
    for i in range(len(pos_array_in)):

        if pos_array_in[i] == pos_list_each[1] :

            return i
    for j in range(o.98, 1.0, 0.001):

        cosinee = (np.dot(pos_array_in[i],pos_list_each[j]))/(np.linalg.norm(pos_array_in[i])*np.linalg.norm(pos_list_each[j])


        if np.all(cosinee) == 1 and i != j:

            periode = np.max(time_list[i] - time_list[j])

    return periode




def apoapsis(pos_list_each):



    return numpy.max(pos_list_each, axis=0)

def pos_list_each(Plist, pos_list, label):


    no_parts = len(Plist)

    for n in range(no_parts):

        if Plist[n].label == label:

            #return n

            pos_list_n = np.array(pos_list[n::no_parts])
            break;
            #print("pos_list_n", pos_list[n::no_parts], "\n\n\n")

    return pos_list_n

def total_energy(pot, kinetic):
    """
    Method to return the total energy of a body
    made up by its kinetic energy and potential energy.

    :param particlei: Particle3D instance
    :param particlej: Particle3D instance
    :param G: constant parameter
    :param vector_R: parameter vector_R from potential
    :param R: parameter R from potential
    :return: total energy of body as a float
    """
    no_parts = len(kinetic)
    pot_each = np.zeros(no_parts)
    for i in range(no_parts):

        pot_each[i] = np.sum(pot[i])


    return np.sum(pot_each + kinetic)

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

    sim_param = open(param_info, "r")

    in_file = open(input_file_name, "r")
    #with open(input_file_name, "r") as in_file:
    #    Plist = Particle3D.extract_data(in_file)
    #    print(args)
    # Set up simulation parameters
    #print(sim_param)
    '''
    dt = float(sim_param[0])
    numstep = int(sim_param[1])
    time = float(sim_param[2])
    '''
    dt, numstep, time = exract_parameters(sim_param)
    # Set up two particles initial conditions and energy from input_file:
    Plist = Particle3D.extract_data(in_file)
    no_parts = len(Plist)
    force_in = force_dw(Plist)
    pos_array_in  = np.array([o.position for o in Plist])
    potential_in = pot_energy_dw(Plist)
    kintic_in = kinetic_energy_dw(Plist)
    energy_in = total_energy(potential_in, kintic_in)
    acceleration_in = acceleration(Plist, force_in)

    for i in range(no_parts):

        Plist[i].velocity = Plist[i].velocity - Particle3D.cm_velocity(Plist)
    
    for i in range(0):

        outfile.write(str(no_parts)+"\n")
        outfile.write("point = %d\r\n" % (i+1))
        for p in Plist:

            if str(p) is not None:

                outfile.write(str(p)+"\n")
    
    # Initialise data lists for plotting later
    time_list = [time]
    pos_list = [o.position for o in Plist]
    energy_list = [energy_in]
    # Start the time integration loop

    for i in range(numstep):
        # Update particle position
        for n in range(no_parts):


            Plist[n].leap_pos2nd(dt, acceleration_in[n,0])

    # Update force
        force_new = force_dw(Plist)
        #print(force_new)
        acceleration_new = acceleration(Plist, force_new)

        for n in range(no_parts):


            # Update particle velocity by averaging acceleration
            # current and new forces
            Plist[n].leap_velocity(dt, 0.5*(acceleration_in[n,0]+acceleration_new[n,0]))


        # Re-define force value
        #Plist = Plist

        acceleration_in = acceleration_new

        # Increase time
        time = time + dt


        potential_new = pot_energy_dw(Plist)
        kintic_new = kinetic_energy_dw(Plist)
        energy_new = total_energy(potential_new, kintic_new)
        

        # Append information to data lists

        time_list.append(time)
        for o in Plist:
            pos_list.append(o.position) 
        energy_list.append(energy_new)

        #write in the output file
    
        outfile.write(str(no_parts)+"\n")
        outfile.write("point = %d\r\n" % (i+1))
        for p in Plist:

            outfile.write(str(p)+"\n")
     
    # Post-simulation:
    #print(pos_list[1000])
    earth = str("Earth")
    pos_list_earth = pos_list_each(Plist, pos_list, earth)
    #period_earth = period(pos_list_earth, time_list)
    #print(pos_list_earth)
    #print(pos_array_in)
    #print(period_earth)
    # Close all files
    outfile.close()
    in_file.close()
    sim_param.close()

    

   # Make vector separation and energy plots
    #make_pyplot(time_list, pos_list, "vector separation")
    #make_pyplot(time_list, energy_list, "Potential Energy")

    # fluctuation for energy
    #mean_en = energy_list[0]
    #max_en = np.max(energy_list)
    #print("The energy fluctuation is {0:6f}".format(math.abs((max_en-mean_en)/mean_en)))


if __name__ == "__main__":

    main(sys.argv[1],sys.argv[2],sys.argv[3])