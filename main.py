import pygame
import neat
import os
import math
import time
import random

pygame.font.init()
WIN_WIDTH = 600
WIN_HEIGHT = 800
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

obstacle = pygame.image.load(os.path.join("images","obstacle.png")).convert_alpha()
mouse = pygame.image.load(os.path.join("images","mouse.png")).convert_alpha()
background = pygame.transform.scale2x(pygame.image.load(os.path.join("images","background.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
clock = pygame.time.Clock()
gen = 0
selecteds = 0
alive = 50

class Mouse:
    rotation = 25
    animation_time = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.speed = 0
        self.base_image = mouse
        self.image = mouse
        self.rotation = 0
        self.rect = self.image.get_rect(center = (round(self.x), round(self.y)))


    def draw(self, win):
        win.blit(self.image, self.rect)

    def turn(self, amount):
        self.rotation += amount
        if self.rotation > 360:
            self.rotation -= 360
        elif self.rotation < 0:
            self.rotation += 360
        self.image = pygame.transform.rotate(self.base_image, self.rotation)
        self.rect = self.image.get_rect(center = (round(self.x), round(self.y)))

    def move(self, move):
        self.x += move * math.cos(math.radians(self.rotation + 90))
        self.y -= move * math.sin(math.radians(self.rotation + 90))
        self.rect = self.image.get_rect(center = (round(self.x), round(self.y)))

    def get_rect(self):
        return self.rect

    def get_mask(self):
        return pygame.mask.from_surface(self.image.convert_alpha())

    def draw_radar(self, win, obstacle_x, obstacle_y):
        pygame.draw.line(win, pygame.Color(0,0,0), (self.x, self.y), (obstacle_x, obstacle_y), width = 2)
        pygame.display.flip()

class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.image = obstacle


    def draw(self, win):
        rotated_image = pygame.transform.rotate(self.image, self.tilt)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x,self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def collide(self, mouse):
        mouse_mask = mouse.get_mask()
        obstacle_mask = pygame.mask.from_surface(self.image)
        obstacle_offset = (self.x - mouse.get_rect().left, self.y - mouse.get_rect().top)
        # Collision flag between mouse and obstacle
        c_point = mouse_mask.overlap(obstacle_mask, obstacle_offset)
        if c_point:
            return True
        return False


def draw_window(win, mouses, obstacles):
    global gen
    global selecteds
    if gen == 0:
         gen = 1
    # generations
    generation_label = STAT_FONT.render("Generation: " + str(gen),1,(200,200,255))
    # alive
    survivors_label = STAT_FONT.render("Alive: " + str(alive),1,(200,200,255))
    win.fill((255,255,255))
    win.blit(generation_label, (1, 30))
    win.blit(survivors_label, (1, 100))
    for mouse in mouses:
        mouse.draw(win)
    for obstacle in obstacles:
        obstacle.draw(win)
    pygame.display.update()



def main(genomes, config):
    global gen
    global selecteds
    global alive
    gen += 1
    networks = []
    ges = []
    mouses = []
    selecteds = 0
    for _, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome,config)
        networks.append(network)
        mouses.append(Mouse(230,700))
        genome.fitness = 0
        ges.append(genome)
    #obstacle2 = Obstacle(400,200)
    obstacles = []
    # for i in range(10):
    #     obstacles.append(Obstacle(random.randint(0,600),random.randint(0,600)))
    obstacles.append(Obstacle(450,250))
    obstacles.append(Obstacle(410,400))
    obstacles.append(Obstacle(250,450))
    obstacles.append(Obstacle(150,600))
    obstacles.append(Obstacle(200,300))


    collision = False
    run = True
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT),1,1)
    collide = False
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        if len(mouses) > 0:
            pass
        else:
            alive = 100
            run = False
        # if not collide:
        #     mouse.turn(0.1)
        #     mouse.move(0.5)
        for x,mouse in enumerate(mouses):
            # Removal of mouses going upside down
            if mouse.rotation > 170 and mouse.rotation < 185:
                ges[x].fitness =- 1
                mouses.pop(x)
                alive -= 1
            # Greeting of mouses going up
            if mouse.rotation < 90 or mouse.rotation > 270:
                ges[x].fitness += 1
            if mouse.y > 10:
                mouse.move(1.8)
                remaining_obstacles = []
                for obs in obstacles:
                    # Adds obstacles still above the mouse
                    if obs.y < mouse.y:
                        remaining_obstacles.append(obs)
                if len(remaining_obstacles) < len(obstacles):
                    # Tells the NN the aim is to go up
                    ges[x].fitness += 1
                if len(remaining_obstacles) > 0:
                    remaining_obstacles.sort(key=lambda x:x.y, reverse=True)
                    # Takes the closest obstacle from the mouse
                    # Depending on y coordinates
                    obstacle = remaining_obstacles[0]
                    # Takes the middle of the obstacle as x coordinate
                    x_obstacle = obstacle.x + (obstacle.image.get_width()/2)
                    # mouse.draw_radar(win, x_obstacle, obstacle.y)
                    # (Pythagorean theorm)
                    next_obstacle = 0
                    x__next_obstacle = 0
                    # Maximum distance possible between an obstacle and a mouse
                    max_distance_obstacle = math.sqrt((WIN_HEIGHT**2) + (WIN_WIDTH**2))
                    # Distance between mouse and obstacle
                    distance_obstacle = math.sqrt((mouse.x - x_obstacle)**2 + (mouse.y - obstacle.y)**2)
                    coeff_obstacle = (mouse.y - obstacle.y)/(mouse.x - obstacle.x)
                    coeff_next_obstacle = 0
                    print(coeff_obstacle)
                    # Initiates the distance of the next_obstacle to
                    distance_next_obstacle = 0
                    if len(remaining_obstacles) > 1:
                        next_obstacle = remaining_obstacles[1]
                        # Takes the middle of the obstacle as x coordinate
                        x_next_obstacle = obstacle.x + (obstacle.image.get_width()/2)
                        distance_next_obstacle = math.sqrt((mouse.x - x_next_obstacle)**2 + (mouse.y - next_obstacle.y)**2)
                        coeff_next_obstacle = (mouse.y - next_obstacle.y)/(mouse.x - next_obstacle.y)
                    # Activation function for obstacle detection
                    # output = networks[x].activate((distance_obstacle/(max_distance_obstacle),
                    #     distance_next_obstacle/(max_distance_obstacle), coeff_obstacle/3 ))
                    output = networks[x].activate((coeff_obstacle,
                        coeff_next_obstacle, distance_obstacle/max_distance_obstacle))
                    # Flag to determine wether to turn right or left
                    sign = 1
                    # Angle used for the rotation
                    angle = 0
                    # Decides wether to turn or not based
                    # On the result of the activation function
                    # Enters the condition if the mouse is too close to the
                    # Current obstacle
                    if output[0] > 0.5 and (mouse.rotation < 90 or mouse.rotation > 270):
                        # If we're at the left of the obstacle we turn left
                        sign = -sign * math.copysign(1,coeff_obstacle)
                        angle = sign * (1 -(1/abs(mouse.x - obstacle.x)))
                    # Else we go up
                    else:
                        ges[x].fitness += 0.5
                        if mouse.rotation > 0 and mouse.rotation < 180:
                            sign = -1
                        else:
                            sign = 1
                        angle = sign * 1
                    mouse.turn(angle)
                else:
                    # depending on th rotation, the mouse will turn left or right
                    # The angle is defined on [0,1]
                    # If there is no more obstacles, the angle will be multiplied by itself
                    # Decreasing the angle to 0, the mouse will go up
                    angle = 0
                    if mouse.rotation > 0 and mouse.rotation < 270:
                        angle = -1
                    else:
                        angle = 1
                    mouse.turn(angle* (1 - (mouse.rotation/(mouse.rotation + 360))))
            # If all the obstacles have been passed, we increase the NN fitness
            else:
                selecteds += 1
                ges[x].fitness += 5

        draw_window(win, mouses, obstacles)
        for obstacle in obstacles:
            for x,mouse in enumerate(mouses):
                if obstacle.collide(mouse) or mouse.x > 590 or mouse.y > 700 or mouse.x < 0:
                    # decreases the fitness score of the neural network
                    # who collided the obstacle
                    ges[x].fitness -= 1
                    mouses.pop(x)
                    networks.pop(x)
                    alive -= 1
                elif mouse.y < 20:
                    mouses.pop(x)
                    ges[x].fitness += 5
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main,50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_feedforward.txt")
    run(config_path)
