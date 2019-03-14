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


def exract_parameters(file_parameters):
    """
    Method to get the parameteres from a the opened file on the molecule O2 or N2
    first line contains parameters for O2 and second line for N2
    
    :param file_parameters: should be already open in main that contains the parameters
    :param molecule: String of the label of the Particle3D.
    :return: The three values D_e, r_e and alpha on O2 or N2 (they are different)
    """
    line = file_parameters.readlines()

    return float(line[0]), int(line[1]), float(line[2])

def force_dw(Plist):
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
    Method to return potential energy of a particle in
    a Morse potential which is given by:
    V(r1,r2) = D_e * ((1-exp[-a*(R12 - re)])**2 - 1)

    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :param D_e: parameter D_e from potential
    :param r_e: parameter r_e from potential
    :param alpha: parameter alpha from potential
    :return: potential energy of particle as float
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


'''
def cm_velocity(Plist):


    no_parts = len(Plist)
    cm_velocity = np.zeros((no_parts, no_parts, 3))
    for i in range(len(Plist)):
        for j in range(i+1, len(Plist)):

            v1 = Particle3D.velocity(Plist[i])
            v2 = Particle3D.velcoity(Plist[j])
            m1 = Particle3D.mass(Plist[i])
            m2 = Particle3D.mass(Plist[j])

            cm_velocity[i] += (v1*m1 + v2*m2)/(m1 + m2)
            cm_velocity[j] -= (v1 * m1 + v2 * m2) / (m1 + m2)

            maybe writing this in particle3D
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
def period(pos_list_each, time_list):


    r = np.zeros(len(pos_list_each))
    min_r_index = np.argmin(r)
    max_r = r[np.argmax(r)]

    for i in range(len(pos_list_each)):

        
        r[i] = np.linalg.norm(pos_list_each[i])


    #print(r[min_r_index])
    #print(r[np.argmax(r)])
    
    #for i in range(len(pos_list_each)):

        #if 
    #print(r)
    #print(np.argmax(r))
    #print(np.argmin(r))

    return 2*abs(time_list[np.argmax(r)] - time_list[np.argmin(r)])




'''

    cosinee = np.zeros(len(pos_list_each))
    periode = np.zeros(len(time_list))
    #print(pos_list_each)
    for i in range(len(pos_array_in)):

        if np.linalg.norm(pos_array_in[i]) == np.linalg.norm(pos_list_each[0]):

            return i

    for j in range(len(pos_list_each)):

        while cosinee < 0.98 :

            cosinee[j] = (np.dot(pos_array_in[i],pos_list_each[j]))/(np.linalg.norm(pos_array_in[i])*np.linalg.norm(pos_list_each[j]))

            if cosinee[j] in range(0.98, 1.0, 0.001) and j > 360:

                return (j)

            else:

                break

    return time_list[j]

'''


def period_moon(pos_list_moon, pos_list_earth, time_list):

    r = np.zeros(len(pos_list_moon))

    for i in range(len(pos_list_moon)):

        
        r[i] = abs(np.linalg.norm(pos_list_earth[i]-pos_list_moon[i]))



    return 2*abs(time_list[np.argmax(r)] - time_list[np.argmin(r)])





def apoapsis(pos_list_each):

    apoapsis = np.zeros(len(pos_list_each))

    for i in range(len(pos_list_each)):

        
        apoapsis[i] = np.linalg.norm(pos_list_each[i])

    return np.max(apoapsis)

def apoapsis_moon(pos_list_moon, pos_list_earth):

    apoapsis_moon = np.zeros(len(pos_list_moon))

    for i in range(len(pos_list_moon)):

        
        apoapsis_moon[i] = abs(np.linalg.norm(pos_list_moon[i]-pos_list_earth[i]))

    return np.max(apoapsis_moon)


def preapsis(pos_list_each):

    preapsis = np.zeros(len(pos_list_each))

    for i in range(len(pos_list_each)):

        
        preapsis[i] = np.linalg.norm(pos_list_each[i])

    return np.min(preapsis)

def preapsis_moon(pos_list_moon, pos_list_earth):

    preapsis_moon = np.zeros(len(pos_list_moon))

    for i in range(len(pos_list_moon)):

        
        preapsis_moon[i] = abs(np.linalg.norm(pos_list_moon[i]-pos_list_earth[i]))

    return np.min(preapsis_moon)

def apsis(apoapsis, preapsis):


    return 0.5*(apoapsis + preapsis)

def pos_list_each(Plist, pos_list, label):


    no_parts = len(Plist)
    pos_list_n = np.zeros((no_parts, 3))

    for n in range(no_parts):

        if Plist[n].label == label:

            #return n

            pos_list_n = np.array(pos_list[n::no_parts])
            break;
            #print("pos_list_n", pos_list[n::no_parts], "\n\n\n")

    return pos_list_n

def total_energy(pot, kinetic):
    """
    Method to return the total energy of the molecule
    (i.e. two particles) made up by their kinetic energies
    plus the potential energy of the bond.

    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :param D_e: parameter D_e from potential
    :param r_e: parameter r_e from potential
    :param alpha: parameter alpha from potential
    :return: total energy of molecule as a float
    """
    no_parts = len(kinetic)
    pot_each = np.zeros(no_parts)
    for i in range(no_parts):

        pot_each[i] = np.sum(pot[i])


    return np.sum(pot_each + kinetic)

# Begin main code
def main(argv1, argv2, argv3):

    # Read name of output file from command line
    if len(sys.argv)!=5:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + "<Particle input>" + "<Param input>" + "<output file>" + "<energy file>")
        quit()
    else:
        energy_file = sys.argv[4]
        outfile_name = sys.argv[3]
        param_info = sys.argv[2]
        input_file_name = sys.argv[1]


    # Open input and output file
    outfile = open(outfile_name, "w")

    energy_out = open(energy_file, "w")

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

        energy_out.write(str(np.sum(potential_new))+","+str(np.sum(kintic_new))+","+str(energy_new)+","+str(time)+"\n")
        outfile.write(str(no_parts)+"\n")
        outfile.write("point = %d\r\n" % (i+1))
        for p in Plist:

            outfile.write(str(p)+"\n")
     
    # Post-simulation:

    Planet = input("Please type the name of the Planet for period and orbit geometry: ")

    if str(Planet) == "Moon":

        pos_list_earth = pos_list_each(Plist, pos_list, str("Earth"))
        pos_list_moon = pos_list_each(Plist, pos_list, str(Planet))
        apoapsis_mooon = apoapsis_moon(pos_list_moon, pos_list_earth)
        preapsis_mooon = preapsis_moon(pos_list_moon, pos_list_earth)
        apsis_moon = apsis(apoapsis_mooon, preapsis_mooon)
        period_mooon = period_moon(pos_list_moon, pos_list_earth, time_list)
        print("apoapsis_moon:" ,apoapsis_mooon, "preapsis_moon:",preapsis_mooon, "apsis_moon:",apsis_moon, "period_moon:", period_mooon)


    elif str(Planet) != "Moon":
        pos_list_Planet = pos_list_each(Plist, pos_list, str(Planet))
        apoapsis_Planet = apoapsis(pos_list_Planet)
        preapsis_Planet = preapsis(pos_list_Planet)
        apsis_Planet = apsis(apoapsis_Planet, preapsis_Planet)
        period_Planet = period(pos_list_Planet, time_list)
        print("apoapsis_Planet:" ,apoapsis_Planet, "preapsis_Planet:",preapsis_Planet, "apsis_Planet:",apsis_Planet, "period_Planet:", period_Planet)

    else:
        print("incorrect label!")

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