# from controller import Supervisor, Robot, Motor
from turtle import speed
from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
from controller import Robot
import struct  # this library is used to interpret the emitter-receiver data

# TIME_STEP = 64
# MAX_SPEED = 6.28


# robot = Robot()

# left_motor = robot.getDevice("left wheel motor")
# left_sensor = robot.getDevice("left wheel sensor")

# right_motor = robot.getDevice("right wheel motor")
# right_sensor = robot.getDevice("right wheel sensor")

# left_motor.setPosition(float('inf'))
# right_motor.setPosition(float('inf'))

# # set up the motor speeds at 10% of the MAX_SPEED.
# left_motor.setVelocity(0.1 * MAX_SPEED)
# right_motor.setVelocity(0.1 * MAX_SPEED)

# while robot.step(TIME_STEP) != -1:
#    pass


# figure out how to do this for every robot

from controller import AnsiCodes
from controller import Robot
# from common import common_print


class SoccerRobot(Robot):  # RobotEmitterReceiverCSV
    def __init__(self):
        """
        The constructor gets the Position Sensor reference and enables it and also initializes the wheels.
        """
        super().__init__()

        # Example name: B1
        self.robot_name = self.getName()
        # B for Blue or Y for Yellow team
        self.robot_team = self.robot_name[0]
        # 1 2 or 3 for the number of the robot player
        self.robot_num = self.robot_name[1]
        # This is the robot node
        self.robot = self.getFromDef(self.robot_name)

        self.timestep = int(self.robot.getBasicTimeStep())

        # Receiver enabling
        self.receiver = self.initialize_comms()

        # Motor enabling
        self.right_motor, self.left_motor = self.setup_motors()
        # Sensor enabling
        self.right_sensor, self.left_sensor = self.setup_sensors()

        # Defining constants
        self.maxSpeed = 10
        self.turn_delta = 0.5

    def initialize_comms(self, receiver_name='receiver'):

        receiver = self.robot.getDevice(receiver_name)
        receiver.setChannel(self.robot_num)
        receiver.enable(self.timestep)

        return receiver

    def setup_motors(self):
        # initializing both wheels
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor = self.robot.getDevice('left wheel motor')

        # setting position and velocity of each wheel
        self.right_motor.setPosition(float('inf'))
        self.right_motor.setVelocity(0.0)

        self.left_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)

    def setup_sensors(self):
        right_sensor = self.robot.getDevice('right wheel sensor')
        left_sensor = self.robot.getDevice('left wheel sensor')

        right_sensor.enable(self.timestep)
        left_sensor.enable(self.timestep)

        return right_sensor, left_sensor

    def create_message(self):
        """
        This method packs the robot's observation into a list of strings to be sent to the supervisor.
        The message contains only the Position Sensor value, ie. the angle from vertical position in radians.
        From Webots documentation:
        'The getValue function returns the most recent value measured by the specified position sensor. Depending on
        the type, it will return a value in radians (angular position sensor) or in meters (linear position sensor).'

        :return: A list of strings with the robot's observations.
        :rtype: list
        """
        message = [self.right_sensor.getValue(),
                   self.left_sensor.getValue()]
        return message

    def handle_receiver(self):
        """
        Modified handle_receiver from the basic implementation of deepbots.
        This one consumes all available messages in the queue during the step it is called.
        """
        while self.receiver.getQueueLength() > 0:
            # Receive and decode message from supervisor
            message = self.receiver.getData().decode("utf-8")
            # Convert string message into a list
            message = message.split(",")
            print('the message is:', message)

            self.use_message_data(message)

            self.receiver.nextPacket()

    # fix this to check what we do with the message
    def use_message_data(self, message):
        """
        0 or else: Do nothing (means : keep going forward)
        1 : Turn right
        2 : Turn left
        """
        # If message is not a list, and is a single action just erase the loop
        # If message is list of actions :

        for action in message:
            if int(action) == 1:
                r_wheel_speed = self.maxSpeed / 2 + self.turn_delta
                l_wheel_speed = self.maxSpeed / 2 - self.turn_delta

            if int(action) == 2:
                r_wheel_speed = self.maxSpeed / 2 - self.turn_delta
                l_wheel_speed = self.maxSpeed / 2 + self.turn_delta

            else:
                continue

        self.right_motor.setVelocity(r_wheel_speed)
        self.left_motor.setVelocity(l_wheel_speed)
        # action = float(message[0])

        # if action:
        #     # we need to change motor speed
        #     # and direction
        #     motorSpeed = 5
        #     pass
        # else:
        #     motorSpeed = 2

        # for i in range(1):
        #     self.right_motor.setPosition(float('inf'))
        #     self.right_motor.setVelocity(motorSpeed)

        #     self.left_motor.setPosition(float('inf'))
        #     self.left_motor.setVelocity(motorSpeed)

    # def handle_emitter(self):
    #     data = self.create_message()
    #     string_message = ''

    #     if type(data) is list:
    #         string_message = ",".join(map(str, data))
    #     elif type(data) is str:
    #         string_message = data
    #     else:
    #         raise TypeError(
    #             'Message must be either a comma-seperated string or a 1-D list'
    #         )
    #     string_message = string_message.encode('utf-8')
    #     self.emitter.send(string_message)

    def run(self):
        while self.robot.step(self.timestep) != 1:
            self.handle_receiver()
            # self.handle_emitter()


robot_controller = SoccerRobot()
robot_controller.run()

# Maybe in the run i need to implement this:
# Need to translate it into python commands
# while(wb_receiver_get_queue_length(receiver) > 0):
#     wb_receiver_get_data(receiver)
#     # we need to save this data somehow and use it here:
#     # ....

#     # this function erases the package that is currently read
#     # And makes the next package the "head" package that needs to be read
#     wb_receiver_next_packet(receiver)

# For the agent's behaviour:
# We have to keep track of the ball's coordinates and compare them to our
# own. Thing is ball doesnt have an emitter to send coordinates so we have to
# Find them somehow
# Position sensor doesn't recognize objects so how do we do it ?

# TO IMPLEMENT:
# # robot_names = ["B1", "B2", "B3", "Y1", "Y2", "Y3"]
# # score = [0, 0]
# # max_time = 10 * 60
# # ball_reset_timer = 0
# # ball_initial_translation = [0, 0, 0]

# # robot_node = robot.getFromDef('B1')

# # robot_nodes = []
# # for name in robot_names:
# #     robot_nodes.append(robot.getFromDef(name))

# # translation_field = robot_nodes[i].getField('translation')

# # new_value = [2.5, 0, 0]
# # translation_field.setSFVec3f(new_value)

# const char *robot_name[ROBOTS] = {"B1", "B2", "B3", "Y1", "Y2", "Y3"};
#   WbNodeRef node;
#   WbFieldRef robot_translation_field[ROBOTS], robot_rotation_field[ROBOTS], ball_translation_field;
#   WbDeviceTag emitter;
#   int i, j;
#   int score[2] = {0, 0};
#   double time = 10 * 60;  // a match lasts for 10 minutes
#   double ball_reset_timer = 0;
#   double ball_initial_translation[3] = {0, 0, 0};
#   double robot_initial_translation[ROBOTS][3] = {{0.3, -0.2, 0.03817},  {0.3, 0.2, 0.03817},  {0.75, 0, 0.03817},
#                                                  {-0.3, -0.2, 0.03817}, {-0.3, 0.2, 0.03817}, {-0.75, 0, 0.03817}};
#   double robot_initial_rotation[ROBOTS][4] = {{0, 0, 1, 1.57}, {0, 0, 1, 1.57}, {0, 0, 1, 3.14},
#                                               {0, 0, 1, 1.57}, {0, 0, 1, 1.57}, {0, 0, 1, -3.14}};
#   double packet[ROBOTS * 3 + 2];
#   char time_string[64];
#   const double *robot_translation[ROBOTS], *robot_rotation[ROBOTS], *ball_translation;
