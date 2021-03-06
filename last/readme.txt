Computer Modeling: Astronomical N-body Simulation
by Mohammadreza Aboutalebi (s1664598) and Austin Morris (s1728541)
******************************************************************

This is a program that will simulate the solar system with N-body objects interacting through Newtonian gravity. This file describes how to operate the program, which is written in Python.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Download and unpack the tarball (compressed archive). The program folder should contain two (.py) files, Particle3D and velocityVerlet3D, as well as four (.txt) files, Param, Particles, Energy, and this readme.

The Param.txt file contains three parameters, dt, number of steps, and start time, which can be adjusted to increase and decrease the simulation accuracy and time span. dt comes set at 0.5, but increase this to calculate longer periods. Pluto has the longest period in the system (248 years ~ 90520 days), which for reasonable program speed and accuracy, we used parameters 
10
9100
0

The Particle.txt file contains the initial conditions and masses of the Planets (+Pluto), Earth's Moon, and Halley's Comet.  The format is label,x,y,z,v_x,v_y,v_z,mass, and the units are in au, au/day, and kg. The Ephemeris data was compiled from NASA/JPL HORIZON (https://ssd.jpl.nasa.gov/horizons.cgi) on 2019-Mar-11 00:00:00.0000.

The Energy.txt file is formatted to print columns of potential energy, kinetic energy, total energy, and time of the system, respectively.

VMD is used to model the system. To observe the Moon's orbit around Earth, the default sphere thickness must be decreased and zoom set to the Earth's frame.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Commands:
[locate program folder]
>python velocityVerlet3D.py Particle.txt Param.txt output.xyz Energy.txt
[after running, input desired object for period calculation, as labeled in Particle.txt]
[e.g.Sun,Mercury,Venus,Earth,Mars,Jupiter,Saturn,Uranus,Neptune,Pluto,Moon,Halley]
>vmd output.xyz