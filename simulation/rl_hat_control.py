""" rl_hat_control.py is the main file for reinforcement learning of 
    heliotropic hat. This version uses a neural net for Q learning.
    
    Author: Jonathon Sather
    Last updated: 1/04/2017 


    TODO: 
           x Make experience replay and fixed batch size.
           x Add "tests" every X iterations and display results to console.
           x Add remove duplicates function. (currently removed)
           x Look more at DeepMind strategies and see if there is something
             else I could implement.
           x Make fxn that creates and saves subplots of loss and performance data
             versus epoch.
           x Make fxn that saves hyper-parameter values.
           x Make fxn that saves nnet parameter values.
           x Make it so that can go between interactive and training modes by
             pressing keys.
           x Fill experience replay before start counting epochs.
           x Look at rewards at each time and see if make sense.
           x Add separate target network (see: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.md8y212iz).
           x Change network architecture to give Q-values for each action all
             at once.
           x Get the damn neural net to do what I want.
           x Add error handling to prompt. (Added try statement to d_key_pressed)
           x Make hat.theta go between 0-2*pi (I think it will work better) ((It does.))
           x Clean it alllll up.
           x Put on Mercurial- see ME507 handout.
"""

from keras.models import load_model
from matplotlib import pyplot
import numpy as np
import pdb
import pygame
import sys
import time
from time import sleep

import hat_model
import rl_hatdisplay
import rl_nnet
import trig_identities

def angle_test(model, hat, environment, angle_light, angle_start=0.01, 
               actions=(1, -1), ambient=0.001, show=False):
    """ Tests if hat can rotate from angle_start to block out direction
        source at angle_light.
    """

    # Set up environment.
    environment.clearSources()
    environment.updateAmbient(ambient)

    dir_phi = 5 * np.pi / 4
    dir_mag = 1.0
    dir_theta = angle_light

    (dirX, dirY, dirZ) = trig_identities.spherical_to_cartesian(dir_mag,
     dir_phi, dir_theta)
    environment.addDirectionalSource((dirX, dirY, dirZ))
    environment.update()

    # Set up hat.
    hat.theta = angle_start

    # Find best angle
    best_angle = find_best_angle(environment, hat)

    # Run test. Passed test if best angle obtained at least twice.
    best_count = 0

    for turn in range(int(2 * np.pi / hat.thetaDot) + 1):
        if hat.theta == best_angle:
            best_count += 1

        # Rotate bill according to best action.
        LS_values = hat.getLSValues()
        angle = np.array([[hat.theta]])
        state = np.transpose(np.hstack((LS_values, angle))) 
        Q = model.run_forward(state.reshape(-1, num_ls + 1))
        a = choose_action(Q)
        hat.rotateBill(a)

        # Update simulation.
        environment.update()

        if show:
            pv.updateHat(hat, 
                         environment.directionalSources[0].direction)
    
    # Return 1 if passed test. 0 otherwise.
    if best_count >= 2:
        return 1
    else:
        return 0

def choose_action(Q, epsilon=0):
    """ Chooses action based on epsilon Q values. """

    # Choose random action wp epsilon *or* if Q values equal.
    if np.random.rand() < epsilon or Q[0][0] == Q[0][1]:
        if np.random.rand() < 0.5:
            a = 1
        else:
            a = -1
    else:
        if max(Q[0][:]) == Q[0][0]:
            a = 1
        else:
            a = -1

    return a

def display_dir_source(dir_mag, dir_phi, dir_theta):
    """ Prints "directional source" prompt (part of interactive mode). """

    print("Directional source:")
    print("  -magnitude: " + str(dir_mag))
    print("  -phi: " + str(dir_phi))
    print("  -theta: " + str(dir_theta))

def display_interactive_prompt():
    """ Prints "interactive mode" main prompt. """
    int_prompt = "Interactive mode!"
    print("\n")
    print(int_prompt.center(70))
    print("  -Press <d> to enter directional source.")
    print("  -Press <r> for random directional theta.")
    print("  -Press <space> or <backspace> to reset bill.")
    print("  -Press <a> to perform learning assessment.")
    print("  -Press <p> to pause.")
    print("  -Use keypad to inc/dec directional source.")
    print("  -Press <q> to quit interactive mode.")
    print("\n")

def d_key_pressed(environment, ambient=0.001):
    """ Takes user input for theta and updates display. Function called
        after 'd' pressed during interactive mode.
    """

    environment.clearSources()
    environment.updateAmbient(ambient)

    dir_mag = 1.0
    dir_phi = 5 * np.pi / 4
    while True:
        try:
            dir_theta = input('Theta: ')
            break
        except:
            print("Not a valid angle!")

    (dirX, dirY, dirZ) = trig_identities.spherical_to_cartesian(dir_mag,
     dir_phi, dir_theta)
    environment.addDirectionalSource((dirX, dirY, dirZ))
    environment.update()

    return environment, dir_mag, dir_phi, dir_theta

                    
def find_best_angle(environment, hat):
    """ Finds angle with least light intensity for given epoch """

    theta = hat.theta # Save original angle 

    # Rotate CCW until at boundary
    rotating = 1
    while rotating:
        rotating = hat.rotateBillCCW()
    
    # Find best angle
    min_intensity = None
    best_angle = None
    rotating = 1
    while rotating:
        environment.update()
        intensity = np.sum(hat.getLSValues())

        if min_intensity == None or intensity < min_intensity:
            min_intensity = intensity
            best_angle = hat.theta

        rotating = hat.rotateBillCW()
    
    hat.theta = theta

    return best_angle

def find_epsilon(epsilon, epoch, update_epsilon, start_updates=499):
    """ Updates epsilon ("random guessing rate") based on epoch. """

    if epsilon <= 0.1:
        epsilon = 0.1
    elif (not (epoch % update_epsilon)) and epoch > start_updates:
        epsilon -= 0.1
    return epsilon

def get_reward(state, best_angle, angle_reward, angle_penalty):
    """ Calculates reward based on state. """

    if state[-1] == best_angle: # Reward if bill at best angle
        reward = angle_reward
    else:
        reward = angle_penalty

    return reward

def interactive_mode(model, hat, environment, deg_per_action=18):
    """ Runs "interactive mode" where the user can update lighting
        conditions and see how the hat reacts.
    """

    # Constants
    actions = (1, -1)               # Set of possible action.
    inc = 0.125                     # Light source direction inc.

    # Variables
    running = True                  # Flag to run simulation.

    # Update rotation speed.
    rotation_speed = deg_per_action / 360. * 2 * np.pi
    hat.updateSpeed(rotation_speed)

    display_interactive_prompt()

    # Set up random directional source
    environment, dir_mag, dir_phi, dir_theta = r_key_pressed(environment)

    while running:
        # Rotate bill according to best action.
        # LS_values = hat.getLSValues()
        # angle = np.array([[hat.theta]])
        # state = np.transpose(np.hstack((LS_values, angle))) 
        state = hat.get_state()

        Q = model.run_forward(state.reshape(-1, num_ls + 1))
        a = choose_action(Q)
        hat.rotateBill(a)

        # Update simulation.
        environment.update()
        pv.updateHat(hat, environment.directionalSources[0].direction)
        sleep(0.05)

        # Handle user input. TODO: Add error checking!
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d: # Manually enter dir light source
                    environment, dir_mag, dir_phi, dir_theta = d_key_pressed(environment)
                    display_dir_source(dir_mag, dir_phi, dir_theta)
                elif event.key == pygame.K_r: # Random light source
                    environment, dir_mag, dir_phi, dir_theta = r_key_pressed(environment)
                    display_dir_source(dir_mag, dir_phi, dir_theta)
                elif event.key == pygame.K_RIGHT: # Rotate light source CW.
                    environment, dir_mag, dir_phi, dir_theta = right_pressed(environment, dir_theta, inc)
                    display_dir_source(dir_mag, dir_phi, dir_theta)
                elif event.key == pygame.K_LEFT: # Rotate light source CCW.
                    environment, dir_mag, dir_phi, dir_theta = left_pressed(environment, dir_theta, inc)
                    display_dir_source(dir_mag, dir_phi, dir_theta)
                elif event.key == pygame.K_SPACE: # Reset bill pos. 0+.
                    hat.theta = 0.01
                elif event.key == pygame.K_BACKSPACE: # Reset bill pos. 0-.
                    hat.theta = 2 * np.pi -0.01
                elif event.key == pygame.K_p: # Pause simulation.
                    print("Simulation paused.")
                    pdb.set_trace()
                elif event.key == pygame.K_a: # Assess hat performance.
                    print("\nAssessing hat performance...")
                    report = test_intelligence(model, hat, environment, show=True)[0]
                    print(report + "\n")
                elif event.key == pygame.K_q: # Quit interacive mode
                    print("\nQuitting interactive mode.\n")
                    running = False

def left_pressed(environment, dir_theta, inc, ambient=0.001):
    """ Rotates theta by value inc CCW. Function called after <left arrow>
        pressed during interactive mode.
    """

    environment.clearSources()
    environment.updateAmbient(ambient)

    update = dir_theta - inc + np.random.normal(scale=0.01 * inc)
    dir_theta = update % (2 * np.pi)

    (dirX, dirY, dirZ) = trig_identities.spherical_to_cartesian(dir_mag,
     dir_phi, dir_theta)
    environment.addDirectionalSource((dirX, dirY, dirZ))
    environment.update()

    return environment, dir_mag, dir_phi, dir_theta

def new_lighting(environment, dir_mag, dir_phi, ambient=0.001):
    """ Sets up new directional source at random theta with specified
        magnitude and phi.
    """

    environment.clearSources()
    environment.updateAmbient(ambient)

    dir_phi = np.random.normal(loc=1.25 * np.pi, scale=0.0625 * np.pi)

    dir_theta = np.random.uniform(0, 2 * np.pi) # Choose random dir_theta.
    (dirX, dirY, dirZ) = trig_identities.spherical_to_cartesian(dir_mag,
                         dir_phi, dir_theta)
    environment.addDirectionalSource((dirX, dirY, dirZ))
    environment.update()

    return dir_theta

def plot_results(errors, performance_log, save=False):
    """ Creates plots of loss and test results up to when function called. """
    
    # Split data into x and y axes for each subplot
    errs_y = [error[0] for error in errors]
    errs_x = [error[1] for error in errors]
    perf_f = [score[0][0] for score in performance_log]
    perf_r = [score[0][1] for score in performance_log]
    perf_x = [score[1] for score in performance_log]
    
    # Create plots
    f, (ax1, ax2) = pyplot.subplots(2, sharex=True)
    ax1.plot(errs_x, errs_y)
    ax1.set_title('Q Network Loss', fontsize=14)
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_xlabel('Epoch')

    ax2.plot(perf_x, perf_f, label='CW')
    ax2.plot(perf_x, perf_r, label='CCW')
    ax2.set_title('Test Results', fontsize=14)
    ax2.set_ylabel('Successful Trials')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left', fontsize='medium')
    ax2.annotate(time.strftime("%m/%d/%Y, %H:%M:%S"), xy=(1, 0),
                    xycoords='axes fraction', fontsize=8,
                    horizontalalignment='right', verticalalignment='bottom')
    
    f.subplots_adjust(hspace=0.5) # Adjust spacing
    fig = pyplot.gcf() 
    fig.canvas.set_window_title('Performance Plots') # Add window title

    if save:
        fig.savefig(time.strftime("../Plots/%m%d%Y_%H%M%S"))

    f.show()


def print_vals(errors, epsilon):
    """ Prints latest error, guessing rate and learning rate to console,
        if possible.
    """

    try:
        print("Error: " + str(errors[-1][0]) + 
          ", Epsilon: " + '{0:05f}'.format(epsilon))
    except IndexError:
        pass

def right_pressed(environment, dir_theta, inc, ambient=0.001):
    """ Rotates theta by value inc CW. Function called after <right arrow>
        pressed during interactive mode.
    """

    environment.clearSources()
    environment.updateAmbient(ambient)

    update = dir_theta + inc + np.random.normal(scale=0.01 * inc)
    dir_theta = update % (2 * np.pi)

    (dirX, dirY, dirZ) = trig_identities.spherical_to_cartesian(dir_mag,
     dir_phi, dir_theta)
    environment.addDirectionalSource((dirX, dirY, dirZ))
    environment.update()

    return environment, dir_mag, dir_phi, dir_theta

def r_key_pressed(environment, ambient=0.001):
    """ Updates theta to random direction between -pi and pi. Function
        called after 'r' pressed during interactive mode.
    """

    environment.clearSources()
    environment.updateAmbient(ambient)

    dir_phi = 5 * np.pi / 4
    dir_mag = 1.0
    dir_theta = np.random.uniform(0, 2 * np.pi)

    (dirX, dirY, dirZ) = trig_identities.spherical_to_cartesian(dir_mag,
     dir_phi, dir_theta)
    environment.addDirectionalSource((dirX, dirY, dirZ))
    environment.update()

    return environment, dir_mag, dir_phi, dir_theta

def save_hyperparams(hyperparams, location):
    """ Logs hyperparameter values to text file. File name current date
        and time.
    """

    with open(location + time.strftime("%m%d%Y_%H%M%S"), 'w') as param_file:
        for param, val in hyperparams.iteritems():
            param_str = param + ": " + str(val) + "\n"
            param_file.write(param_str)

def test_intelligence(model, hat, environment, show=False):
    """ Performs a series of tests to evaluate progress of hat learning.
        Returns string with results.
    """

    # Run angle test at equal intervals in both directions.
    num_tests = 20
    angles = [inc / 10. * np.pi for inc in range(20)]

    f_pass = 0 # Forwards pass counter.
    r_pass = 0 # Reverse pass counter.

    for angle in angles:
        f_pass += angle_test(model, hat, environment, angle, angle_start=0.01,
                             show=show)
        r_pass += angle_test(model, hat, environment, angle, 
                  angle_start=2 * np.pi - 0.01, show=show)

    report = ("Successful forward passes: " + str(f_pass) + " out of " + 
              str(num_tests) + ".\nSuccessful reverse passes: " + str(r_pass) +
              " out of " + str(num_tests) + ".")

    return report, (f_pass, r_pass)


def train_nn_val_approx(main_model, target_model, exp_replay, mini_batch, angle_reward,
                        num_ls=20, actions=(1, -1), numReward=None):
    """ Trains neural network to approximate value function. Draws size 
        'mini_batch' random samples from experience replay to train with.
    """

    # Create mini batch 
    X = []
    Y = [] 

    # Get numReward angle_reward training examples per batch if specified.
    if numReward != None:
       training_samples = exp_replay.remove(mini_batch, random=True, pop=False,
                                            numReward=numReward, reward=10.)
    else:
        training_samples = exp_replay.remove(mini_batch, random=True, pop=False)


    reward_count = 0

    for sample in training_samples:
        Q_old = target_model.run_forward(sample[0].reshape(-1, num_ls + 1))
        Q_new = target_model.run_forward(sample[3].reshape(-1, num_ls + 1))

        y = np.zeros_like(Q_old)
        y[:] = Q_old[:]

        if sample[2] == angle_reward: # No future discount if end of epoch
            update = sample[2]
            reward_count += 1
        else:
            update = sample[2] + gamma * max(Q_new[0][:])

        if sample[1] == actions[0]:
            y[0][0] = update
        else:
            y[0][1] = update

        X.append(sample[0])
        Y.append(y)

    X = np.asarray(X).reshape(-1, num_ls + 1)
    Y = np.asarray(Y).reshape(-1, 2)
    
    # Train with mini batch
    cur_cost = main_model.train_batch(X, Y)

    return cur_cost, main_model

def transfer_weights(main_model, target_model):
    """ Set target model weights to main_model's """
    
    for index, layer in enumerate(main_model.model.layers):
        target_model.model.layers[index].set_weights(layer.get_weights())

    return target_model

def update_target(main_model, target_model, rate=0.01):
    """ Slowly updates target model parameters to main model's. """

    for l_index, layer in enumerate(main_model.model.layers):
        update_weights = []
        for w_index, weight in enumerate(layer.get_weights()):
            old = target_model.model.layers[l_index].get_weights()[w_index]
            update_weights.append(old + rate * (weight - old))
        target_model.model.layers[l_index].set_weights(update_weights)

    return target_model    
        
if __name__ == '__main__':
    # Random seed for predictability
    np.random.seed(8)

    ############################ Constants ####################################
    # Hat and lighting 
    actions = (1, -1)                 # Set of actions. 1 == CW, -1 == CCW
    ambient = 0.001                   # Ambient light component.
    deg_per_action = 18.              # Degrees of bill rotation per action.
    dir_phi = 5. * np.pi / 4          # Dir. light source angle in phi dir.
    dir_mag = 1.0                     # Dir. light source magnitude.
    restricted = True                 # Indicates whether hat rotatation
#                                       has boundary at 0/2*pi.

    # Load/save info
    load_weights = False               # Indicates whether to load in a model.
    model_name_load = "model7"        # Name of model for loading.
    model_name_save = "model7"        # Name of model for saving.
    model_loc = "/media/jsather/PKBACK# 001/Heliotropic Hat/simulation/models/"    
                                      # Location to save/load model.
    note = """ Training model1 with lower learning rate to see if can learn 
               more. 
           """                        # Note for saving.
    train_model = True                # Indicates whether to train or load model.
    var_save = "/media/jsather/PKBACK# 001/Heliotropic Hat/simulation/variables/"
                                      # Location to save variables values in
#                                       text file.

    # Reinforcement learning
    angle_penalty = -.1               # Penalty for not being at best angle. 
    angle_reward = 10.                # Reward for being at best angle.
    epochs = 100000                   # Max epochs for RL model training.
    gamma = 0.9                       # Discount rate.
    learning_rate = 0.00001          # Learning rate for main network.
    max_actions_per_epoch = 100       # Max actions per epoch.
    mini_batch = 100                  # Mini batch size from replay buffer.
    num_ls = 20                       # Number of light sensors on hat.
    replay_size = 5000              # Max number of elements in exp. replay.
    rewards_per_batch = None          # Number of rewarded states in each
#                                       training batch. ("None" == random)
    target_update = 0.001             # Rate at which target model updates.
    tolerance = 0.00001               # Tolerance RL model convergence.

    # Update intervals
    display_interval = 1000           # Number of epochs before disp. updated.
    fill_interval = 1000              # Number of elements filled in exp 
#                                       buffer between status updates.    
    test_interval = 100.              # Number of epochs between testing. 
    train_interval = 4                # Number of steps between training network.
    update_epsilon = epochs / 20.     # Number of epochs before epsilon update.
    vals_interval = 10                # Number of epochs between printings of
#                                       error and epsilon to console.

    ############################# Variables ###################################
    epoch_count = 0                   # Running count of number of epochs.
    epsilon = 1.0                     # Random guessing rate of RL model.
    errors = []                       # CE errors from each training iteration.
    new_batch = True                  # Flag indicating start new batch.
    performance_log = []              # Holds running list of test results.
    show_theta = True                 # Flag indicating to display bill angle
#                                       when displaying epoch.
    show_vals = True                  # Flag indicating to print vals.
    total_steps = 0                   # Holds total number of training steps.

    # Set up environment
    hat = hat_model.Hat((200, 150, 0), 0, 1, restricted=True)
    hat.includeLightSensors(num_ls)
    rotation_speed = deg_per_action / 360. * 2 * np.pi
    hat.updateSpeed(rotation_speed)

    environment = hat_model.Environment((400, 300))
    environment.addHat(hat)

    # Set up visualization
    pv = rl_hatdisplay.ProjectionViewer(environment.boundary[0], environment.boundary[1],
                                        viewing_angle = np.pi / 6)

    # Initialize experience replay
    exp_replay = rl_nnet.experienceReplay(replay_size)
    
    # Initialize main and target networks
    main_model = rl_nnet.QNetwork(input_dim=num_ls+1, lr=learning_rate)
    target_model = rl_nnet.QNetwork(input_dim=num_ls+1, lr=learning_rate)
    target_model = transfer_weights(main_model, target_model)

    if load_weights:
        # Initialize model to load into.
        loaded_model = rl_nnet.QNetwork(input_dim=num_ls+1)

        try:
            loaded_model.new_model(load_model(model_loc + model_name_load))   
        except:
            print("Error loading model! Goodbye.")
            sys.exit()

        # Transfer weights of loaded model into main_model and target_model
        main_model = transfer_weights(loaded_model, main_model)
        target_model = transfer_weights(loaded_model, main_model)

    if train_model:
        # Fill replay buffer
        while not exp_replay.is_full():

            # Initialize new lighting
            dir_theta = new_lighting(environment, dir_mag, dir_phi)

            # Randomly initialize bill position
            hat.theta = np.random.uniform(0, 2 * np.pi)
            environment.update()

            # Find best angle (for reward)
            best_angle = find_best_angle(environment, hat)
            environment.update()

            # Get current hat state
            # LS_values = hat.getLSValues()
            # angle = np.array([[hat.theta]])
            # state = np.transpose(np.hstack((LS_values, angle)))
            state = hat.get_state()

            # Loop through epoch.
            for count in range(max_actions_per_epoch):

                # Choose next action based on Q and epsilon.
                Q = main_model.run_forward(state.reshape(-1, num_ls + 1))
                action = choose_action(Q, epsilon=epsilon) 

                # Take action and get new state.
                hat.rotateBill(action)
                environment.update()

                new_state = hat.get_state()

                # LS_values = hat.getLSValues()
                # angle = np.array([[hat.theta]])
                # new_state = np.transpose(np.hstack((LS_values, angle)))

                # Approximate value at new_state and add to training data.
                reward = get_reward(new_state, best_angle,
                                    angle_reward, angle_penalty)  

                # Add to experience replay and print status if appropriate.
                if not (len(exp_replay.buf) % fill_interval):
                    print("\nFilling experience replay...")
                    print("Buffer contents: " + 
                          str(len(exp_replay.buf)))

                exp_replay.add_tuple((state, action, reward, new_state))
                state = new_state

                if reward == angle_reward:
                    break

        # Train RL model and save parameters.
        for epoch in range(epochs):

            show_theta = True
            show_vals = True

            # Perform test every test_interval epochs and print result.
            if (not (epoch % test_interval) and 
                exp_replay.has_at_least(replay_size - mini_batch)):
                print("\nAssessing hat performance...")
                report, scores = test_intelligence(main_model, hat, environment)
                performance_log.append((scores, epoch))
                print(report)

            # Update parameters
            epsilon =  find_epsilon(epsilon, epoch, update_epsilon)

            # Initialize new lighting
            dir_theta = new_lighting(environment, dir_mag, dir_phi)

            # Randomly initialize bill position
            hat.theta = np.random.uniform(0, 2 * np.pi)
            environment.update()

            # Find best angle (for reward)
            best_angle = find_best_angle(environment, hat)
            environment.update()

            # Get current hat state
            # LS_values = hat.getLSValues()
            # angle = np.array([[hat.theta]])
            # state = np.transpose(np.hstack((LS_values, angle)))
            state = hat.get_state()
            
            # Loop through epoch.
            for count in range(max_actions_per_epoch):

                # Check for key presses
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_g: # Show learning graph
                            print("\nPlotting results.\n")
                            plot_results(errors, performance_log)
                        if event.key == pygame.K_i: # Enter interactive mode
                            interactive_mode(main_model, hat, environment,
                                             deg_per_action=deg_per_action)
                        if event.key == pygame.K_p: # Pause learning
                            print("\nLearning paused.\n")
                            pdb.set_trace()  
                        if event.key == pygame.K_s: # Save model
                            print("\nSaving model...\n")
                            main_model.save_model(model_loc + model_name_save)
                            print("\nModel saved.\n")

                total_steps += 1

                # Choose next action based on Q and epsilon.
                Q = main_model.run_forward(state.reshape(-1, num_ls + 1))
                action = choose_action(Q, epsilon=epsilon) 

                # Take action and get new state.
                hat.rotateBill(action)
                environment.update()

                # LS_values = hat.getLSValues() # Make "get_state fxn" - Now method of hat!
                # angle = np.array([[hat.theta]])
                # new_state = np.transpose(np.hstack((LS_values, angle)))

                new_state = hat.get_state()

                # If replay buffer filled, display actions of one epoch 
                # every display_interval.
                if not (epoch % display_interval):
                    if show_theta:
                        print("\nEpoch: " + str(epoch))
                        print("Light direction is " + 
                               str(dir_theta * 180 / np.pi) + " deg.\n")
                        show_theta = False

                    pv.updateHat(hat, 
                                 environment.directionalSources[0].direction)
                    sleep(0.1)
                
                # Print Error and Epsilon every val_interval.
                if not ((epoch % vals_interval)) and epoch:
                    if show_vals:
                        print_vals(errors, epsilon)

                    show_vals = False
                
                # Approximate value at new_state and add to training data.
                reward = get_reward(new_state, best_angle,
                                    angle_reward, angle_penalty)  

                # Add to experience replay
                exp_replay.add_tuple((state, action, reward, new_state))
               
                state = new_state
                
                # Train every train_interval steps.
                if not (total_steps % train_interval):
                    error, main_model = train_nn_val_approx(main_model, 
                        target_model, exp_replay, mini_batch, angle_reward,
                        num_ls=num_ls, numReward=rewards_per_batch)
                    errors.append((error, epoch))
                    target_model = update_target(main_model, target_model,
                                                 rate=target_update)

                    # Always print first error.
                    if len(errors) == 1:
                        print_vals(errors, epsilon)

                    # Stop training if model has converged.
                    if errors[-1] < tolerance:
                        print("Convergence achieved.") # Set convergence flag here

                if reward == angle_reward:
                    break
            
        # Save trained model
        main_model.save_model(model_loc + model_name_save)

        # Plot errors and save hyperparameters
        plot_results(errors, performance_log, save=True)

        # Save hyperparameter values
        # Create list of hyperparameters
        hyperparams = {'angle penalty': angle_penalty, 
                       'angle reward': angle_reward,
                       'epochs': epochs, 
                       'max actions': max_actions_per_epoch,
                       'batch size': mini_batch,
                       'num ls': num_ls,
                       'replay size': replay_size,
                       'train interval': train_interval,
                       'epsilon update': update_epsilon,
                       'deg_per_action': deg_per_action,
                       'saved_model name': model_name_save,
                       'num_rewards per_batch': rewards_per_batch,
                       'load_model': load_weights,
                       'loaded_model name': model_name_load,
                       'train_model': train_model,
                       'message': note}

        save_hyperparams(hyperparams, var_save)

    # Enter interactive mode once RL model trained.
    interactive_mode(main_model, hat, environment,
                     deg_per_action=deg_per_action)
