""" trig_identities.py contains a collection of useful trig identities. 

    Author: Jonathon Sather
    Last updated: 1/02/2017
"""

import numpy as np

def spherical_to_cartesian(r, theta, phi):
	""" Converts spherical coordinates to cartesian. """
	
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	return (x, y, z)

if __name__ == '__main__':
	pass