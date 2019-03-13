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
    """
    force_dw = np.zeros((len(Plist),6))
    for i in range (len(Plist)):
        for j in range (i+1, len(Plist)):
            vector_R = Particle3D.Vector_Separation(Plist[i],Plist[j])
            R = np.linalg.norm(vector_R)
            m1m2 = Particle3D.mass(Plist[i])*Particle3D.mass(Plist[j])
            force_dw[i, :] += (((1.48818E-34)*m1m2)/R**3)*vector_R
            force_dw[j, :] -= (((1.48818E-34)*m1m2)/R**3)*vector_R
            #check signs above: +,- might be switched i.e. attraction vs. repellence
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
                force_dw[i, j, :] = (((1.48818E-34)*m1m2)/R**3)*vector_R

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
                pot_energy_dw[i, j] = np.zeros(0)

    return pot_energy_dw

def kinetic_energy_dw(Plist):


    no_parts = len(Plist)
    kinetic_energy = np.zeros(no_parts)
    for i in range(no_parts):

        kinetic_energy[i] = Particle3D.kinetic_energy(i)


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
#def period(pos_list_each):













def apsides(pos_list_each):



    return numpy.mean(pos_list_each, axis=0)

def pos_list_each(pos_list):


    pos_list_1 = np.array(pos_list)








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

    in_file = open(input_file_name, "r")
    #with open(input_file_name, "r") as in_file:
    #    Plist = Particle3D.extract_data(in_file)
    #    print(args)
    # Set up simulation parameters
    #print(sim_param)
    dt = float(sim_param[0])
    numstep = int(sim_param[1])
    time = float(sim_param[2])
    
    # Set up two particles initial conditions and energy from input_file:

    Plist = Particle3D.extract_data(in_file)

    no_parts = len(Plist)

    '''
    for i in range(no_parts):

        Plist[i].velocity = Plist[i].velocity - Particle3D.cm_velocity(Plist)
    '''
    '''
    for i in range(numstep):
    
        #counter = 0
        outfile.write("no_parts")
        outfile.write("point = %d\r\n" % (i+1))
        for p in Plist:
            #outfile.write('%s,%8s,%8s,%8s\n' % (xpoints,ypoints,0))
            outfile.write(str(p)+"\n")
        #counter+=1
    '''

    for i in range(0):

        outfile.write("3\n")
        outfile.write("point = %d\r\n" % (i+1))
        for p in Plist:

            if str(p) is not None:

                outfile.write(str(p)+"\n")
    
    '''
    while i, j <= len(Plist) and i != j:
        acceleration[i] = force_dw[i]/mass(Plist[i])
    '''
    force_in = force_dw(Plist)
    acceleration_in = acceleration(Plist, force_in)
    # Initialise data lists for plotting later
    time_list = [time]
    #pos_list = [p1.position, p2.position, p3.position]
    pos_list = [o.position for o in Plist]

    #velocity_list = [o.velocity for o in Plist]
    #energy_list = [energy]
    #print(pos_list)
    # Start the time integration loop

    for i in range(numstep):
        # Update particle position
        for n in range(no_parts):


            Plist[n].leap_pos2nd(dt, acceleration_in[n,0], Particle3D.cm_velocity(Plist))

    # Update force
        force_new = force_dw(Plist)
        acceleration_new = acceleration(Plist, force_new)

        for n in range(no_parts):


            # Update particle velocity by averaging acceleration
            # current and new forces
            Plist[n].leap_velocity(dt, 0.5*(acceleration_in[n,0]+acceleration_new[n,0]), Particle3D.cm_velocity(Plist))
            #Plist[n].velocity = Plist[n].velocity + dt*(0.5*(acceleration[n,0]+acceleration_new[n,0]) - cm_velocities

        #for n in range(no_parts):


        
        #pos_list.append(o.position for o in Plist)
        #Plist.append(Plist[i])


            #Plist[n].velocity = Plist[n].velocity - Particle3D.cm_velocity(Plist)
        # Re-define force value
        Plist = Plist

        acceleration_in = acceleration_new

        # Increase time
        time = time + dt
        

        # Append information to data lists

        time_list.append(time)
        pos_list.append([o.position for o in Plist])

        #print(pos_list)

        # write in the output file
    
        outfile.write("3\n")
        outfile.write("point = %d\r\n" % (i+1))
        for p in Plist:
            #print(str(p)
            outfile.write(str(p)+"\n")
     
    # Post-simulation:
    
    # Close all files
    #outfile.close()
    #in_file.close()
    #sim_param.close()

    

   # Make vector separation and energy plots
    #make_pyplot(time_list, pos_list, "vector separation")
    #make_pyplot(time_list, energy_list, "Potential Energy")

    # fluctuation for energy
    #mean_en = energy_list[0]
    #max_en = np.max(energy_list)
    #print("The energy fluctuation is {0:6f}".format(math.abs((max_en-mean_en)/mean_en)))


if __name__ == "__main__":

    main(sys.argv[1],sys.argv[2],sys.argv[3])