import time
import turtle
turtle.speed(0)
turtle.delay(0)
turtle.ht()

w = 1000
h = 1000

# Size of cage
vw = w // 2
vh = h // 2

# "Actual" drawn size of cage
cw = vw // 2
ch = vh // 2

turtle.Screen().setup(w, h)

tracer_interval = 100
# Upper bounds for movements on a given timestep
# set to 512 to disable normalization
max_mag = 5
#max_angle = 36  # So we move 10% of a full rotation at most
#max_angle = 720 # Since this empirically is very small
max_angle = 512
#max_angle = 720 # Since this empirically is very small

wall_buffer = 20

# Set to 360 to disable heading checking entirely
#heading_buffer = 45
heading_buffer = 360


class Body(turtle.Turtle):
    def __init__(self, animate=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed(0)
        #self.ht()

        # todo add support for this?
        self.animate = animate

        # Add cage display to the canvas
        self.canvas = turtle.getcanvas()
        self.sense_cage = True
        self.enforce_cage = True
        self.cage = self.canvas.create_rectangle(-cw,ch,cw,-ch, width=3, outline="red")

        self.left_trim = .56474 #Determined by running for 100k timesteps and getting average difference between left and right power
        self.right_trim = 0

        self.lefts = []
        self.rights = []

        turtle.tracer(tracer_interval, 0)
        # self.ht() # hide body

    def cagecolor(self, color):
        self.canvas.itemconfig(self.cage, outline=color)

    def nose_touching(self):
        """
        Determine if its nose is touching a wall, and return True/False depending.
        In order for this to be the case, it has to be:
            1. Within 5 units of a wall
            2. "Facing" a wall, meaning more than 45 degrees of it's heading is
                towards a wall, such that if it were to move forward it would
                be moving towards the wall.

        :return: True/False on if nose touch sensors should be triggered.
        """

        if not self.sense_cage: return False

        # For my own convenience I set x1,y1 to bot-left corner and x2,y2 to top-right
        x1,y1,x2,y2 = -cw,-ch,cw,ch
        pos_x, pos_y = self.pos()
        heading = self.heading()
        dist_2d = lambda x1,x2: abs(x1-x2)

        # Check each wall's case

        # right wall, 315deg - 45deg centered on 0deg or 360deg
        # have to compare both 0 and 360 since circles
        if dist_2d(pos_x,x2) < wall_buffer and dist_2d(heading, 0) < heading_buffer and dist_2d(heading, 360) < heading_buffer:
            return True

        # top wall, 45deg - 135deg centered on 90deg
        if dist_2d(pos_y, y2) < wall_buffer and dist_2d(heading, 90) < heading_buffer:
            return True

        # left wall, 135deg - 225deg centered on 180deg
        if dist_2d(pos_x, x1) < wall_buffer and dist_2d(heading, 180) < heading_buffer:
            return True

        # bottom wall, 225deg - 315deg centered on 270deg
        if dist_2d(pos_y, y1) < wall_buffer and dist_2d(heading, 270) < heading_buffer:
            return True

        # Otherwise, nope!
        return False




    def exit(self):
        if not self.animate:
            turtle.update()
            time.sleep(3)

    def normalize(self, angle, mag):
        """
        We have chosen our max angle and magnitude values at initialization, so we normalize
            left and right to ensure they fit those ranges
            max(angle) = 512 (255 - (-255))
            max(mag) = 512 (255 + 255)

        Note: if 255 ends up being not the max, this needs to change

        So to get normalized values:
            angle* = (angle/max(angle))*max_angle
            mag* = (mag/max(mag))*max_mag


        :param angle: [0,512]
        :param mag: [0,512]
        :return: [normalized_angle in range [0, max_angle], normalized_mag in range [0, max_mag]
        """
        return (angle / 512.) * max_angle, (mag / 512.) * max_mag

    def move(self, left, right):
        """
        Given neuron firings for force to apply to left muscles and right muscles,
            compute the normalized angle and magnitude to move the nematode body
            and move it. Return normalized angle and magnitude.

        Each movement is a vector, in that it has angle and magnitude.
        Given left, right, both in ranges [-255,255]:
            angle = max(left,right) - min(left,right)
            mag = left + right

        With both left/right and forward/backward being determined by whichever is largest.


        :param left: [-255, 255] accumulated left muscle neuron values
        :param right: [-255, 255] accumulated right muscle neuron values
        :return: (angle, mag) indicating the vector we moved
            (normalized for our maximum allowed angle and magnitude)
        """

        #self.lefts.append(left)
        #self.rights.append(right)

        # Add Trims
        left += self.left_trim
        right += self.right_trim

        # Get raw angles and normalize
        angle = max(left, right) - min(left, right)
        mag = left + right
        angle, mag = self.normalize(angle, mag)

        # Rotate first
        if left >= right:
            # More power in the left, so we turn right
            self.right(angle)
        else:
            # More power in the right, so we turn left
            self.left(angle)

        # Move next
        if mag >= 0:
            self.forward(mag)
        else:
            self.backward(-mag)

        if self.enforce_cage:
            # Keep it bounded to box, enforce by resetting position to boundaries
            # For my own convenience I set x1,y1 to bot-left corner and x2,y2 to top-right
            x1,y1,x2,y2 = -cw,-ch,cw,ch
            pos_x, pos_y = self.pos()

            if pos_x < x1:
                pos_x = x1
            elif pos_x > x2:
                pos_x = x2

            if pos_y < y1:
                pos_y = y1
            elif pos_y > y2:
                pos_y = y2

            # If not outside the box, no change, if outside the box, reset to the box bounds
            self.setpos(pos_x,pos_y)

        return angle, mag
