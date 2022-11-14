import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import matplotlib.animation as animation
from numpy import sin, cos


# Functions for the double pendulum simulation:

def simulate_double_pendulum(length1, length2, mass1, mass2, theta1=120.0, theta2=-10.0, w1=0.0, w2=0.0):
    """
    Simulates the double pendulum. Code is based on the example code at
    https://matplotlib.org/stable/gallery/animation/double_pendulum.html

    Parameters
    ----------
    length1 : float
        The length of the first rod. 
    length2 : float
       The length of the second rod. 
    mass1 : float
       The mass of the first bob. 
    mass2 : float
        The mass of the second bob.
    theta1: float
        The initial angle of the first rod
    theta2: float
        The initial angle of the second rod    
    w1: float
        The initial angular velocity of the first rod
    w2: float
        The initial angular velocity of the second rod
        
    Returns the simulation, a plot of the angular velocity vs. the angle of both 
    of the bobs and a plot of the angle of both of the bobs vs. time.

    """
    
    # The gravitational acceleration
    g=9.81
    
    # Set the current state
    current_state = np.radians([theta1, w1, theta2, w2])
    
    
    # Function to calculate the derivative
    def calculate_derivatives(current_state, t):
        """
        Calculates the derivatives to be used in calculating the next point of 
        the pendulum. Equations from https://matplotlib.org/stable/gallery/animation/double_pendulum.html
        
        Parameters
        ----------
        current_state : array
            Current state of the system. 
        t : float
            Current time.

        Returns
        -------
        derivatives : array
            Array of the derivatives of the angles of both of the pendulums.

        """
        
        # Create an array to store the values 
        derivatives = np.zeros_like(current_state)
        
        # First value of the array is the w1, which is also the first derivative of
        # theta1
        derivatives[0] = current_state[1]

        # To simplify the equations, set delta = theta2 - theta1
        delta = current_state[2] - current_state[0]
        
        # Again, to simplify the equations let's assign a variable for another part 
        # of the function
        den1 = (mass1+mass2) * length1 - mass2 * length1 * cos(delta)**2
        
        # The second derivative for the first angle:
        derivatives[1] = ((mass2 * length1 * current_state[1]**2 * sin(delta) * cos(delta)
                    + mass2 * g * sin(current_state[2]) * cos(delta)
                    + mass2 * length2 * current_state[3]**2 * sin(delta)
                    - (mass1+mass2) * g * sin(current_state[0]))
                   / den1)
        
        # The third value of the array is the w2, which is also the first derivative of
        # theta2
        derivatives[2] = current_state[3]
        
        # Assign another variable to simplify the equation
        den2 = (length2/length1) * den1
        
        # The second derivative of the second angle
        derivatives[3] = ((- mass2 * length2 * current_state[3]**2 * sin(delta) * cos(delta)
                    + (mass1+mass2) * g * sin(current_state[0]) * cos(delta)
                    - (mass1+mass2) * length1 * current_state[1]**2 * sin(delta)
                    - (mass1+mass2) * g * sin(current_state[2]))
                   / den2)

        return derivatives
    
    
    # create a time array sampled at 0.05 second steps
    dt = 0.05
    t = np.arange(0, 20, dt)

    # Solve the ordinary differential equations by integrating them
    y = integrate.odeint(calculate_derivatives, current_state, t)
    
    # Coordinates for the first bob
    x1 = length1*sin(y[:, 0])
    y1 = -length1*cos(y[:, 0])
    
    # Coordinates for the second bob
    x2 = length2*sin(y[:, 2]) + x1
    y2 = -length2*cos(y[:, 2]) + y1
    

    # Commands to set the figure
    fig = plt.figure()
    ax = plt.axes(xlim=(-(length1+length2), (length1+length2)), ylim=(-(1+length1+length2), length1+length2))
    ax.set_aspect('equal')
    ax.grid()

    # Plot the figure
    line, = ax.plot(0, 0, 'o-', lw=2, color='black')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.6, 0.9, '', transform=ax.transAxes)
    
    # Function to animate the double pendulum
    def draw_double_pendulum(i):
        """
        Draws the movement of the double pendulum. This function is used in
        the animation. Part of the code is from https://scipython.com/blog/the-double-pendulum/
        
        Parameters
        ----------
        i : integer
            Index of the frame to be drawn

        """
        
        # Set the data to be drawn
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        
        # Plot a trail of the second bob's route (this part is from 
        # https://scipython.com/blog/the-double-pendulum/ )
        
        trail_secs = 4
        # This corresponds to max_trail time points.
        max_trail = int(trail_secs / dt)
        
        # The trail will be divided into ns segments
        ns = 20
        s = max_trail // ns
        
        # Draw the trail
        for j in range(ns):
            imin = i - (ns-j)*s
            if imin < 0:
                continue
            imax = imin + s + 1

            # Plot the trail
            ax.plot(x2[imin:imax], y2[imin:imax], c='b', lw=0.7)
            
        return line, time_text
    
    # Animate the system
    motion = animation.FuncAnimation(fig, draw_double_pendulum, range(1, len(y)),
                              interval=dt*1000, blit=True)
    motion.save("double_pendulum.gif")
    plt.show()
 
    # Idea for the following plots is from 
    # https://adamdempsey90.github.io/python/double_pendulum/double_pendulum.html   
 
    # Plot the angular velocity of both of the bobs versus the angle of the bobs.
    plt.plot(y[:,0], y[:,1], 'o', label="first bob")
    plt.plot(y[:,2], y[:,3], 'o', label="second bob")
    plt.xlabel("angle")
    plt.ylabel("angular velocity")
    plt.legend()
    plt.show()    
    
    # Plot the angle of the first and second bob versus time.
    plt.plot(t, y[:,0], lw=2, label="first bob")
    plt.plot(t, y[:,2], lw=2, label="second bob")
    plt.xlabel("time")
    plt.ylabel("angle")
    plt.legend()
    plt.show()
    
    
# Functions for the Lorenz attractor simulation:
    
def simulate_lorenz_attractor(x_initial, y_initial, z_initial):
    """
    Simulates the Lorenz attractor. Function is based on the example at
    https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

    Parameters
    ----------
    x_initial : float
        Initial value for the x-coordinate.
    y_initial : float
        Initial value for the y-coordinate.
    z_initial : float
        Initial value for the z-coordinate.

    Returns the 3D simulation of the system and 2D plots in x-y, x-z and y-z 
    planes.

    """
   
    
    def lorenz_equations(x, y, z, sigma=10, rho=28, beta=8.0/3.0):
        """
        
        Calculates the Lorenz equations.
        
        Parameters:
           x, y, z: coordinates for a point in a three dimensional space
           sigma, rho, beta: Lorenz attractor parameters
        Returns:
           dx_dt, dy_dt, dz_dt: values of the Lorenz attractor's 
               derivatives at coordinates x, y, z
        """
        dx_dt = sigma*(y - x)
        dy_dt = x*(rho - z) - y 
        dz_dt = x*y - beta*z
        return dx_dt, dy_dt, dz_dt
    
    
    # Values for the number of time points and the interval between them
    dt = 0.01
    num_steps = 5000

    # Create lists to store the values of the coordinates in every point
    x_list = np.empty(num_steps + 1)
    y_list = np.empty(num_steps + 1)
    z_list = np.empty(num_steps + 1)

    # Set initial values
    x_list[0], y_list[0], z_list[0] = x_initial, y_initial, z_initial

    # Calculate the partial derivatives at the current point and use them 
    # to estimate the next point. 
    for i in range(num_steps):
        dx_dt, dy_dt, dz_dt = lorenz_equations(x_list[i], y_list[i], z_list[i])
        x_list[i + 1] = x_list[i] + (dx_dt * dt)
        y_list[i + 1] = y_list[i] + (dy_dt * dt)
        z_list[i + 1] = z_list[i] + (dz_dt * dt)
        
    
    # Set the figure
    fig = plt.figure(figsize=(12, 9))
    
    # Set the axis limits and projection
    
    # The following command setting the axis can be unchecked to make the animation 
    # work, but then you have to put # before the next command
    # ax = fig.gca(projection='3d') 
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim((-25,25))
    ax.set_ylim((-25,25))
    ax.set_zlim((0,50))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    line, = plt.plot([],[], color="blue", lw=0.7)
    
    
    def draw_lorenz(num):
        """
        Draw the Lorenz attractor one frame at a time.

        Function is used to animate the system.
        
        Parameters
        ----------
        num : integer
            Index of the time step.

        """
        # Plot the line one step at a time and make the simulation go faster 
        # by adding more points in one animation frame
        line.set_data(x_list[:num*30],y_list[:num*30])
        
        line.set_3d_properties(z_list[:num*30])
        
        fig.canvas.draw()
        
        return line,
    
    # Animate the Lorenz function and save it as a gif.
    motion = animation.FuncAnimation(fig, draw_lorenz, range(1, num_steps), 
                              interval=20*dt, blit=True)
    motion.save("lorenz.gif")
    plt.show()
    
    
    #Plot 2-dimensional x-y projection of the attractor
    plt.plot(x_list[:],y_list[:])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    # Plot 2-dimensional x-z projection of the attractor
    plt.plot(x_list[:],z_list[:], color='green')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.show()

    # Plot 2-dimensional y-z projection of the attractor
    plt.plot(y_list[:],z_list[:], color='violet')
    plt.xlabel("y")
    plt.ylabel("z")
    plt.show()

def main(): 
    """
    The main function.
    
    Returns either or both of the simulations.

    """
    
    # Choose which simulation you want to do
    simulate_double_pendulum(2.0, 2.0, 2.0, 2.0, 140.0, 210.0, 0.0, 0.0)
    #simulate_lorenz_attractor(1.0, 1.0, 1.5)
 
    
# Run the main function    
main()

