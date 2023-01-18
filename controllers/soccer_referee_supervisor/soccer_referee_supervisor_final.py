from deepbots.supervisor.controllers.supervisor_env import SupervisorEnv
import numpy as np
from controller import Supervisor
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange

#field_length_x = 0.7, -0.7
#field_width_y =0.6 , -0.6
# orientation angle: ?
# z i dont know what im supposed to do about that:
#z = 0.0375 - 0.0372

# Action space: 2 times gia to velocity stis rodes
# Maybe action space: discrete: go forward, rotate a bit
# Maybe go forward always, rotate a bit left, rotate a bit right Discrete action space

# Observation space: position_translation , rotation translation, ball_position, ball_rotation


class SoccerSupervisor(SupervisorEnv):

    def __init__(self, num_robots=6):
        super().__init__()

        self.num_robots = num_robots
        # self.steering_degree = 0.5  # how much right or left the robot should "steer"
        self.timestep = int(self.getBasicTimeStep())

        # IMPLEMENT TIME:
        # IMPLEMENT SCORE:
        # IMPLEMENT TEAMS:
        # IMPLEMENT GAME: (goal, end when time ends etc)

        # Environment specific
        # [robot_x, robot_y, robot_orientation(z)] x 6 (robot_num) + x,y pos and z orientation of ball = 21 and we need the ball x and y and orientation(z)
        # [[3], [3], [3], [3], [3], [3], [3]]
        self.observationSpace = 21

        # example: [0,1,1,0,2,0] 1 action for each robot, range: 0-2
        self.actionSpace = 3  # 0 : do nothing, 1 : go left, 2 : go right
        # Find min max values for each one of these variables

        # getting all the robot nodes
        self.robot_names = ["B1", "B2", "B3", "Y1", "Y2", "Y3"]
        self.robot = [self.getFromDef(name) for name in self.robot_names]
        self.robot_size = 0.075

        # supervisor robot:
        self.supervisor = self.getFromDef('robot')

        # Ball:
        self.ball = self.getFromDef('soccer ball')
        self.ball_radius = 0.021

        # GAME:
        self.goal_x_limit = 0.745
        self.game_time = 10*60  # 10 minutes (we deal with seconds)

        # initial positions and rotations
        self.initPositions = [self.robot[i].getField(
            "translation").getSFVec3f() for i in range(self.num_robots)]
        self.initRotations = [self.robot[i].getField(
            "rotation").getSFVec3f() for i in range(self.num_robots)]

        print("initial positions are:", self.initPositions)
        print('initial rotations are:', self.initRotations)

        # Enable communication
        self.communication = self.initialize_comms()

        # Enable wheels and speed
        # self.robot_motors = [self.setup_motors(node) for node in self.robot]

        # Enable sensors
        # self.robot_sensors = [self.setup_sensors(node) for node in self.robot]

        # Variable to save the messages received from the robots
        self.messageReceived = None
        # Score accumulated during an episode
        self.episodeScore = 0
        # A list to save all the episode scores, used to check if task is solved
        self.episodeScoreList = []
        self.test = False                               # Whether the agent is in test mode

    def initialize_comms(self):
        # communication = []
        # for i in range(self.num_robots):
        #     emitter = self.getDevice('emitter')
        #     receiver = self.getDevice('receiver')

        #     emitter.setChannel(i)
        #     receiver.setChannel(i)

        #     receiver.enable(self.timestep)

        #     communication.append({
        #         'emitter': emitter,
        #         'receiver': receiver
        #     })
        communication = []
        for i in range(self.num_robots):
            emitter = self.getDevice('emitter')
            emitter.setChannel(i)
            communication.append(emitter)
        return communication

    def step(self, action):
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        self.handle_emitter(action)

        return(
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info()
        )

    def handle_emitter(self, actions):

        for i, action in enumerate(actions):
            message = str(action).encode('utf-8')
            self.communication[i].send(
                message)

    # def handle_receiver(self, actions):

    #     messages = []
    #     for com in self.communication:
    #         receiver = com['receiver']
    #         if receiver.getQueueLength() > 0:
    #             messages.append(receiver.getData().decode('utf-8'))
    #             receiver.nextPacket()  # not sure if this is legit
    #         else:
    #             messages.append(None)

    #     print('the messages are:', messages)
    #     return messages
    def get_observations(self):

        robotPosition_x = [normalizeToRange(self.robot[i].getPosition(
        )[0], -0.7, 0.7, -1.0, 1.0) for i in range(self.num_robots)]

        robotPosition_y = [normalizeToRange(self.robot[i].getPosition(
        )[1], -0.6, 0.6, -1.0, 1.0) for i in range(self.num_robots)]

        # robotRotation_x = [normalizeToRange(self.robot[i].getOrientation(

        # need to normalize to range(but I dont know the range)
        # Need to find out which value exactly we need
        robotRotation_z = [self.robot[i].getOrientation()[3]
                           for i in range(self.num_robots)]

        robotVelocity = [normalizeToRange(self.robot[i].getVelocity(
        )[2], -10, 10, -1.0, 1.0, clip=True) for i in range(self.num_robots)]  # dont know if this is legit

        ball_obs = self.ball.getPosition()[:1].append(
            self.ball.getOrientation()[2])
        ball_obs.append(self.ball.getVelocity())

        observations = [None for _ in range(self.num_robots)]
        for i in range(self.num_robots):
            observations[i] = [robotPosition_x[i],
                               robotPosition_y[i], robotRotation_z[i]]

        # Add ball observation to the Observations
        observations.append(ball_obs)

        print(observations)
        return observations

    def get_all_observations(self):

        robotPosition_x = [normalizeToRange(self.robot[i].getPosition(
        )[0], -0.7, 0.7, -1.0, 1.0) for i in range(self.num_robots)]

        robotPosition_y = [normalizeToRange(self.robot[i].getPosition(
        )[1], -0.6, 0.6, -1.0, 1.0) for i in range(self.num_robots)]

        # robotRotation_x = [normalizeToRange(self.robot[i].getOrientation(

        # need to normalize to range(but I dont know the range)
        # Need to find out which value exactly we need
        robotRotation_z = [self.robot[i].getOrientation()[3]
                           for i in range(self.num_robots)]

        robotVelocity = [normalizeToRange(self.robot[i].getVelocity(
        )[2], -10, 10, -1.0, 1.0, clip=True) for i in range(self.num_robots)]  # dont know if this is legit

        ball_obs = self.ball.getPosition()[:1].append(
            self.ball.getOrientation()[2])
        ball_obs.append(self.ball.getVelocity())

        observations = [None for _ in range(self.num_robots)]
        for i in range(self.num_robots):
            observations[i] = [robotPosition_x[i],
                               robotPosition_y[i], robotRotation_z[i]]

        # Add ball observation to the Observations
        observations.append(ball_obs)

        print(observations)
        return observations

    def get_reward(self, action=None):
        # figure out the proper reward for the experiment
        # maybe +20 for goal, -20 for goal taken, +1 for ball touch etc
        # maybe even constant -1 for 10 timesteps etc(to give them motive to get a goal)
        reward = None

        return reward

    def is_done(self):
        # An episode is done if the goal difference is greater than 2
        # Or if timelimit ends

        pass

    def get_default_observation(self):
        # not legit need to figure it out more
        observation = []

        for _ in range(self.num_robots):
            robot_obs = [0.0 for _ in range(self.observationSpace)]
            observation.append(robot_obs)

        return observation

    def get_info(self):
        pass

    def solved(self):
        # if last 100 episodes my bot's goal dif is higher than 2 then solved
        pass

    def reset(self):
        self.simulationReset()
        self.simulationResetPhysics()

        # i dont know why i do that, especially 2 times
        super(Supervisor, self).step(int(self.getBasicTimestep()))
        super(Supervisor, self).step(int(self.getBasicTimestep()))

        for i in range(self.num_robots):
            self.communication[i]['receiver'].disable()
            # I dont know why i do that
            self.communication[i]['receiver'].enable(self.timestep)

            # I dont know why i do that
            receiver = self.communication[i]['receiver']
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        return self.get_default_observation()
