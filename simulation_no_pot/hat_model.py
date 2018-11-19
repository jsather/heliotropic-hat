""" hat_model.py models the environment and components for heliotropic hat 
    in 3D. 

    Author: Jonathon Sather
    Last updated: 4/15/2017
"""

import numpy as np
import pdb

class DirectionalSource:
    """ Directional light source w/ direction and intensity. """

    def __init__(self, (dirX, dirY, dirZ)):
        self.direction = np.array([dirX, dirY, dirZ])
        self.intensity = np.linalg.norm(self.direction)

class PointSource:
    """ Point light source w/ position and intensity. """

    def __init__(self, (x, y, z), intensity):
        self.position = np.array([x, y, z])
        self.intensity = intensity

class LightSensor:
    """ Light sensor w/ value proportional to light intensity. """

    def __init__(self, (x, y, z), (dirX, dirY, dirZ)):
        self.position = np.array([x, y, z])
        self.orientation = (np.array([dirX, dirY, dirZ]) /
         np.linalg.norm(np.array([dirX, dirY, dirZ]))) # Ensure unit magnitude
        self.value = 0
        self.color = np.array([255, 255, 0])

class Hat:
    """ A standard baseball hat with spacial orientation and bill position.
        Bill is represented as circular w/ offset from center. Hat class also
        stores light sensors attached to hat as a list of light sensor objects.
    """

    def __init__(self, (x, y, z), theta, scale, restricted=True):
        self.billDiam = 1.5 * scale
        self.billOffset = 0.9 * scale
        self.color = np.array([0, 0, 255]) 
        self.diam = 2 * scale
        self.lightSensors = []
        self.position = np.array([x, y, z])
        self.restricted = restricted    # Restricted bill rotation
        self.theta = theta
        self.thetaDot = 2 * np.pi / 100 # Constant rotational speed

    def getBillCenter(self):
        """ Method to find the center of the bill based on angular position
            and hat geometry.
        """

        center = self.position + np.array([np.cos(self.theta) *
                 self.billOffset, np.sin(self.theta) * self.billOffset, 0])
        return center


    def getLSValues(self, mode='no_noise'):
        """ Fetches values of light sensors currently attached to hat. """

        num_ls = len(self.lightSensors)
        ls_vals = np.empty((1, num_ls))

        for ls in range(num_ls):
            ls_vals[0,ls] = self.lightSensors[ls].value

        if mode == 'add_noise':
            ls_vals += np.random.normal(scale=0.00001,size=(1,num_ls))

        return ls_vals

    def get_state(self, pot=True, actions=None):
        """ Returns current state of hat (light sensor values + bill angle in
            numpy array).
        """
        
        if pot: # State w/ potentiometer
            state = np.transpose(np.hstack((self.getLSValues(),
                                 np.array([[self.theta]]))))
        else:   # State w/ last actions
            state = np.transpose(np.hstack((self.getLSValues(),
                                 np.array([actions]))))
        return state

    def includeLightSensors(self, quantity):
        """ Adds <quantity> light sensors evenly spaced around hat base. 
            Note that this method gets rid of any previously included 
            light sensors.
        """

        self.lightSensors = []
        sectorAngle = 2 * np.pi / quantity

        for i in range(quantity):
            (dx, dy, dz) = (np.cos(sectorAngle * i), np.sin(sectorAngle *
             i), 0)
            (x, y, z) = (self.position[0] + dx, self.position[1] + dy,
                         self.position[2] + dz)

            self.lightSensors.append(LightSensor((x, y, z), (dx, dy, dz)))

    def rotateBill(self, direction):
        """ Rotates bill based on input direction and angular velocity. """
        
        update = self.theta + direction * self.thetaDot
        if self.restricted:
            if update < 0 or update > (2 * np.pi):  # Boundary at 0/2*pi
                pass
            else:                                   # No boundary issues
                self.theta = update
        else:
            self.theta = update % (2 * np.pi)

    def rotateBillCCW(self):
        """ Rotates bill counter-clockwise. Includes stop at 0 radians by
            default. Return 1 if non-restricted rotation. 0 if restricted.
        """

        update = self.theta - self.thetaDot
        rotation = 1

        if self.restricted:
            if update < 0:
                rotation = 0
            else:
                self.theta = update
        else:
            self.theta = self.theta % (2 * np.pi)

        return rotation

    def rotateBillCW(self):
        """ Rotates bill clockwise. Includes stop at 0 radians by default.
            Return 1 if non-restricted rotation. 0 if restricted.
        """

        update = self.theta + self.thetaDot
        rotation = 1

        if self.restricted:
            if update > (2 * np.pi):   # Overflow pi to -pi
                rotation = 0
            else:                                          
                self.theta = update
        else:
            self.theta = update % (2 * np.pi)

        return rotation


    def updateSpeed(self, thetaDot):
        """ Method to update self.thetadot. """

        self.thetaDot = thetaDot

class Environment: 
    """ Environment contains data and regarding the hat environment. """

    def __init__(self, (width, height)):
        self.boundary = np.array([width, height])
        self.color = np.array([255, 255, 255])

        self.hat = None
        self.directionalSources = []
        self.pointSources = []
        self.ambient = 0

    def addHat(self, hat):
        """ Method for adding or replacing hat. """

        self.hat = hat

    def updateColor(self, (r, g, b)):
        """ Updates the color of the environment. """

        self.color = np.array([r, g, b])

    def updateAmbient(self, ambient):
        """ Updates the ambient light value of the environment. """

        self.ambient = ambient

    def addDirectionalSource(self, (dirX, dirY, dirZ)):
        """ Adds a directional source to the environment. """

        self.directionalSources.append(DirectionalSource((dirX, dirY, dirZ)))

    def addPointSource(self, (x, y, z), intensity):
        """ Adds a point light source to the environment. """

        self.pointSources.append(PointSource((x, y, z), intensity))

    def clearSources(self):
        """ Clears the ambient, point and directional light sources from 
            the environment.
        """

        self.directionalSources = []
        self.pointSources = []
        self.ambient = 0

    def notObstructedDirectional(self, hat, lightSensor, directionalSource):
        """ Returns 0 if light sensor obstructed from directional source by
            hat bill or base.
        """
        distance = np.linalg.norm(lightSensor.position - hat.getBillCenter())

        # Only account for bill if light directed downwards.
        if np.dot(directionalSource.direction, np.array([0,0,1])) < 0:
            direction_down = 1
        else:
            direction_down = 0

        if (((distance < hat.billDiam / 2) and direction_down) or
         (np.dot(lightSensor.orientation, directionalSource.direction) > 0)):
            return 0
        
        return 1

    def notObstructedPoint(self, hat, lightSensor, pointSource):
        """ Returns 0 if light sensor obstructed from point source by hat
            bill or base.
        """

        distance = np.linalg.norm(lightSensor.position - hat.getBillCenter())

        dir = lightSensor.position - pointSource.position

        # Only account for bill if light directed downwards.
        if np.dot(dir, np.array([0, 0, 1])) < 0:
            direction_down = 1
        else:
            direction_down = 0
  
        if (((distance < hat.billDiam / 2) and direction_down) or
         (np.dot(lightSensor.orientation, dir) > 0)):
            return 0

        return 1

    def updateLightSensor(self, lightSensor):
        """ Updates a single light sensor's value, considering ambient, point,
            and directional components.
        """

        # Ambient
        lightSensor.value = self.ambient

        # Point
        for p in self.pointSources:
            #pdb.set_trace()
            inv_square = (1 / 
             (np.linalg.norm(p.position - lightSensor.position) ** 2))

            lightSensor.value +=  (
             inv_square * (self.notObstructedPoint(self.hat, 
             lightSensor, p) * np.dot(lightSensor.orientation, - 
             (lightSensor.position - p.position) * p.intensity / 
             np.linalg.norm(lightSensor.position - p.position))))

        # Directional
        for d in self.directionalSources:
            lightSensor.value += (self.notObstructedDirectional(self.hat, 
             lightSensor, d) * np.dot(lightSensor.orientation, - d.direction))

    def update(self):
        """ Updates all light sensors on the hat. """

        for lightSensor in self.hat.lightSensors:
            self.updateLightSensor(lightSensor)

if __name__ == '__main__':
    pass


