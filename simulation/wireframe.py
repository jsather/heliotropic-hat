""" wireframe.py contains functions basic wireframe data to be used
    in Pygame simulation. 

    Author: Jonathon Sather
    Last updated: 11/24/2016
    Lots of this code adapted from: 
        http://www.petercollingridge.co.uk/pygame-physics-simulation
"""

import numpy as np

def translationMatrix(dx=0, dy=0, dz=0):
	""" Return matrix for translation along vector (dx, dy, dz). """

	return np.array([[1, 0, 0, 0],
					 [0, 1, 0, 0],
					 [0, 0, 1, 0],
					 [dx, dy, dz, 1]])

def scaleMatrix(sx=0, sy=0, sz=0):
	""" Return matrix for scaling equally along all axes centered
	 on pt (cx, cy, cz). """

	return np.array([[sx, 0, 0, 0],
					 [0, sy, 0, 0],
					 [0, 0, sz, 0],
					 [0, 0, 0, 1]])

def rotateXMatrix(radians):
	""" Return matrix for rotating about the x-axis by 'radians' rads. """

	c = np.cos(radians)
	s = np.sin(radians)
	return np.array([[1, 0, 0, 0],
					 [0, c, -s, 0],
					 [0, s, c, 0],
					 [0, 0, 0, 1]])

def rotateYMatrix(radians):
	""" Return matrix for rotating about the y-axis by 'radians' rads. """

	c = np.cos(radians)
	s = np.sin(radians)
	return np.array([[c, 0, s, 0],
					 [0, 1, 0, 0],
					 [-s, 0, c, 0],
					 [0, 0, 0, 1]])

def rotateZMatrix(radians):
	""" Return matrix for rotating about the z-axis by 'radians' rads. """

	c = np.cos(radians)
	s = np.sin(radians)
	return np.array([[c, -s, 0, 0],
					 [s, c, 0, 0],
					 [0, 0, 1, 0],
					 [0, 0, 0, 1]])

class Node:
	""" Class that stores coordinates of given node. """

	def __init__(self, coordinates):
		self.x = coordinates[0]
		self.y = coordinates[1]
		self.z = coordinates[2]

class Wireframe:
	""" Class that stores information about a given wireframe, including all
	    the nodes, edges and relevant methods.
	"""

	def __init__(self):
		self.nodes = np.zeros((0, 4))
		self.edges = []

	def addNodes(self, node_array):
		""" Adds an array of nodes to wireframe. """

		ones_column = np.ones((len(node_array), 1))
		ones_added = np.hstack((node_array, ones_column))
		self.nodes = np.vstack((self.nodes, ones_added))

	def addEdges(self, edgeList):
		""" Adds a list of edges to wireframe. """

		self.edges += edgeList

	def clearEdges(self):
		""" Clears edge list from wireframe memberdata. """

		self.edges = []

	def transform(self, matrix):
		""" Applies transformation to wireframe nodes. """

		self.nodes = np.dot(self.nodes, matrix)

	def outputNodes(self):
		""" Prints nodes of wireframe to console. """

		for i in range(self.nodes.shape[0]):
			(x, y, z, _) = self.nodes[i, :]
			print "Node %d: (%.3f, %.3f, %.3f)" % (i, x, y, z)

	def outputEdges(self):
		""" Prints edges of wireframe to console. """

		for i, (start, stop) in enumerate(self.edges):
			node1 = self.nodes[start, :]
			node2 = self.nodes[stop, :]
			print "Edge %d: (%.3f, %.3f, %.3f)" % (i, node1[0], node1[1],
			 node1[2]),
			print "to (%.3f, %.3f, %.3f)" % (node2[0], node2[1], node2[2])

	def findCenter(self):
		""" Finds coordinates of centroid of wireframe. """

		num_nodes = len(self.nodes)
		meanX = sum([node[0] for node in self.nodes]) / num_nodes
		meanY = sum([node[1] for node in self.nodes]) / num_nodes
		meanZ = sum([node[2] for node in self.nodes]) / num_nodes

		return np.array([meanX, meanY, meanZ, 1])

	def translate(self, axis, d):
		""" Add constant 'd' to the coordinate 'axis' of each node of a
			wireframe """

		if axis in ['x', 'y', 'z']:
			for node in self.nodes:
				setattr(node, axis, getattr(node, axis) + d)

	def scale(self, (center_x, center_y), scale):
		""" Scales the wireframe from the center of the screen. """

		for node in self.nodes:
			node[0] = center_x + scale * (node[0]- center_x)
			node[1] = center_y + scale * (node[1] - center_y)
			node[2] *= scale

	def make_circle(self, center, radius, num_nodes, arc_length=2 * np.pi,
	 add_to_wf=True):
		""" Creates wireframe circle in xy plane. """

		center = np.hstack((center, 0))
		angle = arc_length / num_nodes
		nodes = np.zeros((0, 4))
		edges = []
		offset = self.nodes.shape[0]

		for i in range(num_nodes):
			node = np.array([radius, 0, 0, 1])
			rotationMatrix = rotateYMatrix(angle * i)
			node = np.dot(node, rotationMatrix)
			nodes = np.vstack((nodes, node + center))

			if i + 1 != num_nodes:
				edges.append((offset + i, offset + i + 1))


			elif arc_length == 2 * np.pi:
				edges.append((offset + i, offset))

		if add_to_wf == True:		
			self.nodes = np.vstack((self.nodes, nodes))
			self.edges += edges

		return (nodes, edges)

	def make_sphere(self, center, radius, num_nodes_per_hemi, arc_length=
	 2 * np.pi, num_hemis=20, num_lat=21, add_to_wf=True):
		""" Creates wireframe sphere. """

		center = np.hstack((center, 0))

		# Longitude lines
		angle = 2 * np.pi / num_hemis
		nodes = np.zeros((0, 4))
		edges = []

		(circle_nodes, circle_edges) = self.make_circle(np.array([0, 0, 0]), radius,
		 num_nodes_per_hemi, arc_length, False)

		rotate90X = rotateXMatrix(0.5 * np.pi)
		rotate90Z = rotateZMatrix(1 * np.pi)
		circle_nodes = np.dot(circle_nodes, rotate90X)
		circle_nodes = np.dot(circle_nodes, rotate90Z)

		for i in range(num_hemis):
			rotationMatrix = rotateYMatrix(angle * i)
			nodes = np.vstack((nodes, np.dot(circle_nodes, rotationMatrix)))
			edges += [(x + len(self.nodes) + num_nodes_per_hemi * i, y + + len(self.nodes) + num_nodes_per_hemi * i)
			 for (x,y) in circle_edges]

		# Latitudine lines
		lowest_y = -radius * np.sin((arc_length - np.pi) / 2)
		vert_distance = (radius - lowest_y) / num_lat

		(unit_circle_nodes, unit_circle_edges) = self.make_circle(np.array([0, 0, 0]), 1,
		 num_nodes_per_hemi, 2 * np.pi, False)

		for i in range(num_lat):
			# Translate
			new_nodes = -1 * unit_circle_nodes
			new_nodes[:,1] += (lowest_y + vert_distance * i)

			# Scale
			rad = np.sqrt(radius ** 2 - (lowest_y + vert_distance * i) ** 2)
			scalingMatrix = scaleMatrix(rad, 1, rad)
			nodes = np.vstack((nodes, np.dot(-new_nodes, scalingMatrix)))
			edges += [(x + num_nodes_per_hemi * (num_hemis + i), y +
			 num_nodes_per_hemi * (num_hemis + i)) for (x, y) in
			 unit_circle_edges]

		nodes = nodes + center

		if add_to_wf == True:		
			self.nodes = np.vstack((self.nodes, nodes))
			self.edges += edges

		return (nodes, edges)

	def make_dense_circle(self, center, radius, num_nodes, arc_length=2 * np.pi,
	 num_circles=10, add_to_wf=True):
		""" Creates wireframe circle that is 'filled in'. """

		nodes = np.zeros((0, 4))
		edges = []
		scale = radius / num_circles

		(unit_circle_nodes, unit_circle_edges) = self.make_circle(
		 np.array([0, 0, 0]), 1, num_nodes, arc_length, False)
		rotate90Y = rotateYMatrix(-0.5 * np.pi)
		unit_circle_nodes = np.dot(unit_circle_nodes, rotate90Y)

		for i in range(num_circles):
			scalingMatrix = scaleMatrix(scale * (i + 1), 1, scale * (i + 1))
			nodes = np.vstack((nodes, np.dot(1 * unit_circle_nodes, scalingMatrix)))
			edges += [(x + num_nodes * i, y + num_nodes * i)
			 for (x,y) in unit_circle_edges]

		nodes = nodes + np.hstack((center, 0))

		if add_to_wf == True:		
			self.nodes = np.vstack((self.nodes, nodes))
			self.edges += edges

		return (nodes, edges)

class WireframeHat:
	""" Wireframehat contains all the wireframes needed to make a wirefame
	    hat: a base, bill and light sensors.
	"""

	def __init__(self, base, bill, lightSensors=None):
		self.base = base
		self.bill = bill
		self.lightSensors = lightSensors

if __name__ == "__main__":
	# Cube test
	cube_nodes = [(x, y, z) for x in (0,1) for y in (0,1) for z in (0,1)]
	print(cube_nodes)
	
	cube = Wireframe()
	cube.addNodes(np.array(cube_nodes))

	cube.addEdges([(n, n + 4) for n in range(0,4)])
	cube.addEdges([(n, n + 1) for n in range(0, 8, 2)])
	cube.addEdges([(n, n + 2) for n in (0, 1, 4, 5)])

	cube.outputNodes()
	cube.outputEdges()
	
	cube.make_circle(np.array([0, 0, 0]), 1, 8)
	cube.outputNodes()
	cube.outputEdges()