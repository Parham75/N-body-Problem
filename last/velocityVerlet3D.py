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
    :param planet/moon/comet: String of the label of Particle3D.
    :return: The three parameter values dt, number of steps, and total time
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
    """
    Method to return acceleration of a body in
    a Gravitational potential which is given by:
    -(G*m1m2)/R)

    :param particlei: Particle3D instance
    :param particlej: Particle3D instance
    :return: acceleration of body as a float
    """

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
    :return: potential energy of body as a float
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
    """
    Method to return kinetic energy, which is given by:
    1/2*m*v^2
    :param particlei: Particle3D instance
    :return: kinetic energy of body as a float
    """


    no_parts = len(Plist)
    kinetic_energy = np.zeros(no_parts)
    for i in range(no_parts):

        kinetic_energy[i] = Particle3D.kinetic_energy(Plist[i])

    return kinetic_energy


def period(pos_list_each, time_list):
    r = np.zeros(len(pos_list_each))
    min_r = r[np.argmin(r)]
    max_r = r[np.argmax(r)]
    min_r_index = np.argmin(r)

    for i in range(len(pos_list_each)):
        r[i] = np.linalg.norm(pos_list_each[i])

    for i, pi in enumerate(r):

        if i

    return 2 * abs(time_list[np.argmax(r)] - time_list[np.argmin(r)])


def period_moon(pos_list_moon, pos_list_earth, time_list):
    r = np.zeros(len(pos_list_moon))

    for i in range(len(pos_list_moon)):
        r[i] = abs(np.linalg.norm(pos_list_earth[i] - pos_list_moon[i]))

    return 2 * abs(time_list[np.argmax(r)] - time_list[np.argmin(r)])

def apoapsis(pos_list_each):
    apoapsis = np.zeros(len(pos_list_each))

    for i in range(len(pos_list_each)):
        apoapsis[i] = np.linalg.norm(pos_list_each[i])

    return np.max(apoapsis)


def apoapsis_moon(pos_list_moon, pos_list_earth):
    apoapsis_moon = np.zeros(len(pos_list_moon))

    for i in range(len(pos_list_moon)):
        apoapsis_moon[i] = abs(np.linalg.norm(pos_list_moon[i] - pos_list_earth[i]))

    return np.max(apoapsis_moon)


def preapsis(pos_list_each):
    preapsis = np.zeros(len(pos_list_each))

    for i in range(len(pos_list_each)):
        preapsis[i] = np.linalg.norm(pos_list_each[i])

    return np.min(preapsis)


def preapsis_moon(pos_list_moon, pos_list_earth):
    preapsis_moon = np.zeros(len(pos_list_moon))

    for i in range(len(pos_list_moon)):
        preapsis_moon[i] = abs(np.linalg.norm(pos_list_moon[i] - pos_list_earth[i]))

    return np.min(preapsis_moon)


def apsis(apoapsis, preapsis):
    return 0.5 * (apoapsis + preapsis)


def pos_list_each(Plist, pos_list, label):
    no_parts = len(Plist)
    pos_list_n = np.zeros((no_parts, 3))

    for n in range(no_parts):

        if Plist[n].label == label:
            # return n

            pos_list_n = np.array(pos_list[n::no_parts])
            break;
            # print("pos_list_n", pos_list[n::no_parts], "\n\n\n")

    return pos_list_n

def total_energy(pot, kinetic):
    """
    Method to return the total energy of a body,
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
    if len(sys.argv) != 5:
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

    # Set up simulation parameters
    dt, numstep, time = exract_parameters(sim_param)

    # Set up two particles initial conditions and energy from input_file
    Plist = Particle3D.extract_data(in_file)
    no_parts = len(Plist)
    force_in = force_dw(Plist)
    pos_array_in = np.array([o.position for o in Plist])
    potential_in = pot_energy_dw(Plist)
    kintic_in = kinetic_energy_dw(Plist)
    energy_in = total_energy(potential_in, kintic_in)
    acceleration_in = acceleration(Plist, force_in)

    #center of mass correction
    for i in range(no_parts):
        Plist[i].velocity = Plist[i].velocity - Particle3D.cm_velocity(Plist)

    for i in range(0):

        outfile.write(str(no_parts) + "\n")
        outfile.write("point = %d\r\n" % (i + 1))
        for p in Plist:

            if str(p) is not None:
                outfile.write(str(p) + "\n")

    # Initialise data lists
    time_list = [time]
    pos_list = [o.position for o in Plist]
    energy_list = [energy_in]

    # Start the time integration loop
    for i in range(numstep):
        # Update particle position
        for n in range(no_parts):
            Plist[n].leap_pos2nd(dt, acceleration_in[n, 0])

        # Update force
        force_new = force_dw(Plist)
        acceleration_new = acceleration(Plist, force_new)

        for n in range(no_parts):
            # Update particle velocity by averaging acceleration
            # current and new forces
            Plist[n].leap_velocity(dt, 0.5 * (acceleration_in[n, 0] + acceleration_new[n, 0]))

        # Re-define force value

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

        # write in the output file

        energy_out.write(
            str(np.sum(potential_new)) + "," + str(np.sum(kintic_new)) + "," + str(energy_new) + "," + str(time) + "\n")
        outfile.write(str(no_parts) + "\n")
        outfile.write("point = %d\r\n" % (i + 1))
        for p in Plist:
            outfile.write(str(p) + "\n")

    # Post-simulation

    Planet = input("Please type the name of the Planet for period and orbit geometry: ")

    if str(Planet) == "Moon":

        pos_list_earth = pos_list_each(Plist, pos_list, str("Earth"))
        pos_list_moon = pos_list_each(Plist, pos_list, str(Planet))
        apoapsis_mooon = apoapsis_moon(pos_list_moon, pos_list_earth)
        preapsis_mooon = preapsis_moon(pos_list_moon, pos_list_earth)
        apsis_moon = apsis(apoapsis_mooon, preapsis_mooon)
        period_mooon = period_moon(pos_list_moon, pos_list_earth, time_list)
        print(
        "apoapsis_moon:", apoapsis_mooon, "preapsis_moon:", preapsis_mooon, "apsis_moon:", apsis_moon, "period_moon:",
        period_mooon)


    elif str(Planet) != "Moon" and str(Planet) != "Sun":
        pos_list_Planet = pos_list_each(Plist, pos_list, str(Planet))
        apoapsis_Planet = apoapsis(pos_list_Planet)
        preapsis_Planet = preapsis(pos_list_Planet)
        apsis_Planet = apsis(apoapsis_Planet, preapsis_Planet)
        period_Planet = period(pos_list_Planet, time_list)
        print("apoapsis_Planet:", apoapsis_Planet, "preapsis_Planet:", preapsis_Planet, "apsis_Planet:", apsis_Planet,
              "period_Planet:", period_Planet)

    else:
        print("incorrect label!")

    # Close all files
    outfile.close()
    in_file.close()
    sim_param.close()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])