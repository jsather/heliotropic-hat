""" rl_hatdisplay.py builds on wire_frame.py by adding more functions
    for displaying wireframe hat using Pygame.

    Author: Jonathon Sather
    Last updated: 1/03/2017
"""

import numpy as np
import pdb
import pygame

import hat_model
import wireframe as wf

# Matrix to adjust wireframe coordinates to transformations.
correction_matrix = np.eye(4)

def make_wf_hat(hat, pv, nodes_per_hemi = 100):
    """ Makes wireframe hat consisting of base, bill and light sensors.
    """

    global correction_matrix
    
    scale = 50
    
    # Make untransformed base, bill and lightSensor wireframes.
    base = wf.Wireframe()
    bill = wf.Wireframe()
    lightSensors = wf.Wireframe()

    hat_center = hat.position
    bill_center = 1 * hat_center
    ls_center = 1 * hat_center
    ls_center[1] += scale

    bill_center[0] += hat.billOffset * scale * np.cos(hat.theta)
    bill_center[2] += hat.billOffset * scale * np.sin(hat.theta)

    bill_radius = hat.billDiam / 2 * scale
    base_radius = hat.diam / 2 * scale

    lightSensors.make_circle(ls_center, base_radius, len(hat.lightSensors))
    lightSensors.clearEdges()
    
    bill.make_dense_circle(bill_center, bill_radius, nodes_per_hemi)

    bill.edges[:] = [x for x in bill.edges if (np.linalg.norm(bill.nodes[x[0]][0:3]
     - hat_center) > base_radius and np.linalg.norm(bill.nodes[x[1]][0:3]
     - hat_center) > base_radius)]

    base.make_sphere(hat_center, base_radius, nodes_per_hemi, 1 * np.pi, 20, 11)

    # Transform wireframes to viewing angle. 
    center = np.hstack((hat_center, 1))
    base.nodes = base.nodes - center
    bill.nodes = bill.nodes - center
    lightSensors.nodes = lightSensors.nodes - center

    rotateXMatrix = wf.rotateXMatrix(pv.viewing_angle)
    rotateYMatrix = wf.rotateYMatrix(pv.viewing_angle)

    rotateMatrix = np.dot(rotateYMatrix, rotateXMatrix)
    
    rotateMatrix = 1 * rotateXMatrix

    # Update correction matrix.
    correction_matrix = np.dot(np.linalg.inv(rotateMatrix), correction_matrix)

    bill.nodes = np.dot(bill.nodes, rotateMatrix)
    base.nodes = np.dot(base.nodes, rotateMatrix)
    lightSensors.nodes = np.dot(lightSensors.nodes, rotateMatrix)

    bill.nodes = bill.nodes + center
    base.nodes = base.nodes + center
    lightSensors.nodes = lightSensors.nodes + center

    return wf.WireframeHat(base, bill, lightSensors)

class ProjectionViewer:
    """ Displays 3D objects on a Pygame screen. """

    def __init__(self, width, height, viewing_angle = 0):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))

        pygame.display.set_caption('Heliotropic Hat')
        pygame.font.init()
        
        self.background = (10, 10, 50)
        self.viewing_angle = viewing_angle

        self.wireframes = {}
        self.hat = None
        self.thetaDot = 2 * np.pi / 20
        self.directional_sources = {}
        self.point_sources = {}
        self.ambient = 0

        self.displayNodes = True
        self.displayEdges = True
        self.nodeColor = (255, 255, 255)
        self.edgeColor = (200, 200, 200)
        self.nodeRadius = 4

    def run(self):
        """ Creates a Pygame screen until it is closed. """

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        self.rotate_bill(self.thetaDot)

                    elif event.key == pygame.K_p:
                        self.rotate_bill(-self.thetaDot)

            self.display()
            pygame.display.flip()

    def updateHat(self, hat, dir_source):
        """ Overrides old wireframe hat with new one that reflects current
            state, and displays it.
        """
        
        # Make hat and add to memberdata.
        wf_hat = make_wf_hat(hat, self, nodes_per_hemi = 100)
        self.addHat(wf_hat)

        # Update display.
        self.displayNodes = False
        ls_vals = hat.getLSValues()
        self.display(ls_vals, dir_source)
        pygame.display.flip()

    def addWireframe(self, name, wireframe):
        """ Add a named wireframe object. """

        self.wireframes[name] = wireframe

    def addHat(self, hat):
        """ Method for adding or replacing hat. """

        self.hat = hat

    def display(self, ls_vals, dir_source, max_intensity=1):
        """ Draw the wireframes on the screen. """

        self.screen.fill(self.background)
        
        # Display light sensors.
        for wireframe in [self.hat.lightSensors]:
            for index, node in enumerate(wireframe.nodes):

                color = (int(ls_vals[0, index] / max_intensity * 255),
                         int(ls_vals[0, index] / max_intensity * 255), 0)
                
                # Make sure valid color argument.
                for comp in color:       
                    if comp > 255:
                        comp = 255

                pygame.draw.circle(self.screen, color,
                 (int(node[0]), int(node[1])), self.nodeRadius, 0)

        # Display base and bill.
        for wireframe in [self.hat.base, self.hat.bill]:
            if self.displayEdges:
                for n1, n2 in wireframe.edges:
                    pygame.draw.aaline(self.screen, self.edgeColor,
                     wireframe.nodes[n1][:2], wireframe.nodes[n2][:2], 1)

            if self.displayNodes:
                for node in wireframe.nodes:
                    pygame.draw.circle(self.screen, self.nodeColor,
                     (int(node[0]), int(node[1])), self.nodeRadius, 0)

    def translateAll(self, vector):
        """ Translate all wireframes along a given axis by d units. """

        matrix = wf.translationMatrix(*vector)

        for wireframe in [self.hat.base, self.hat.bill, self.hat.lightSensors]:
            wireframe.transform(matrix)

    def scaleAll(self, scale):
        """ Scale all wireframes by given scale, centered at center of screen. """

        center_x = self.width / 2
        center_y = self.height / 2

        for wireframe in [self.hat.base, self.hat.bill, self.hat.lightSensors]:
            wireframe.scale((center_x, center_y), scale)

    def rotateAll(self, axis, theta):
        """ Rotates all wireframe objects in simulation. """

        global correction_matrix
        
        # Create rotation matrix, and update global correction matrix.
        if axis == 'X':
            rotateMatrix = wf.rotateXMatrix(theta)
            correction_matrix = np.dot(wf.rotateXMatrix(-theta),
             correction_matrix)

        elif axis == 'Y':
            rotateMatrix = wf.rotateYMatrix(theta)
            correction_matrix = np.dot(wf.rotateYMatrix(-theta),
             correction_matrix)

        elif axis == 'Z':
            rotateMatrix = wf.rotateZMatrix(theta)
            correction_matrix = np.dot(wf.rotateZMatrix(-theta),
             correction_matrix)

        for wireframe in [self.hat.base, self.hat.bill, self.hat.lightSensors]:
            center = hat.base.findCenter()
            wireframe.nodes = wireframe.nodes - center 
            wireframe.transform(rotateMatrix)
            wireframe.nodes = wireframe.nodes + center 

    def rotate_bill(self, angle):
        """ Rotates bill of hat in simulation. """

        global correction_matrix
        
        # Rotate bill in relative coordinates.
        center = self.hat.base.findCenter()
        hat.bill.nodes = self.hat.bill.nodes - center
        rotateY = wf.rotateYMatrix(angle)

        # Rotate relative to correction matrix.
        self.hat.bill.nodes = np.dot(self.hat.bill.nodes, correction_matrix)
        self.hat.bill.nodes = np.dot(self.hat.bill.nodes, rotateY)
        self.hat.bill.nodes = np.dot(self.hat.bill.nodes,
         np.linalg.inv(correction_matrix))
        self.hat.bill.nodes = self.hat.bill.nodes + center

if __name__ == '__main__':
    pass

