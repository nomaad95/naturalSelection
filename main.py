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
    score_label = STAT_FONT.render("Gens: " + str("test"),1,(200,200,255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = STAT_FONT.render("Generation: " + str(gen),1,(200,200,255))
    win.blit(background, (0,0))
    win.blit(score_label, (1, 30))
    for mouse in mouses:
        mouse.draw(win)
    for obstacle in obstacles:
        obstacle.draw(win)
    pygame.display.update()



def main(genomes, config):
    global gen
    global selecteds
    gen += 1
    networks = []
    ges = []
    mouses = []
    selecteds = 0
    for _, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome,config)
        networks.append(network)
        mouses.append(Mouse(200,700))
        genome.fitness = 0
        ges.append(genome)
    #obstacle2 = Obstacle(400,200)
    obstacles = []
    # for i in range(10):
    #     obstacles.append(Obstacle(random.randint(0,600),random.randint(0,600)))
    obstacles.append(Obstacle(150,600))
    obstacles.append(Obstacle(300,300))
    # obstacles.append(Obstacle(500,100))


    collision = False
    run = True
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    collide = False
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        if len(mouses) > 0:
            pass
        else:
            run = False
            print("quit")
            break
        # if not collide:
        #     mouse.turn(0.1)
        #     mouse.move(0.5)
        for x,mouse in enumerate(mouses):
            if mouse.y > 10:
                remaining_obstacles = []
                for obstacle in obstacles:
                    # adds obstacles still above the mouse
                    if obstacle.y < mouse.y:
                        remaining_obstacles.append(obstacle)
                print(len(remaining_obstacles))
                obstacle = obstacles[0]
                smallest_distance = abs(mouse.y - obstacle.y)
                # Define the smallest distance by the distance between the mouse
                # And the first obstacle
                # As the mouse is moving fast, sometimes a passed obstacle can still
                # be the current obstacle selected.
                # Hence, abs() is used to prevent from any negative values
                for obstacle_loop in remaining_obstacles:
                    if mouse.y - obstacle_loop.y < smallest_distance and (mouse.y > obstacle_loop.y) and (mouse.y - obstacle_loop.y >= 0):
                        print("changement obstacle")
                        obstacle = obstacle_loop
                        smallest_distance = mouse.y - obstacle.y
                mouse.move(1.7)
                ges[x].fitness += 0.1
                #output = networks[x].activate(mouse.x,abs(mouse.x - obstacles[0].x), mouse.y, abs(mouse.y - obstacles[0].y))
                # activation function for obstacle detection
                if len(remaining_obstacles) > 0:
                    output = networks[x].activate((mouse.x, abs(mouse.x - obstacle.x), mouse.y))
                    output2 = networks[x].activate((mouse.y,mouse.y - obstacle.y, mouse.y))
                    sign = 1
                    angle = 0
                    if output[0] > 0.5:
                        if mouse.y > obstacle.y:
                            if(mouse.x - obstacle.x) < 0:
                                sign = 1
                            else:
                                sign = -1
                            angle = sign * (1 -(1/abs(mouse.x - obstacle.x + 0.01)))
                    elif output2[0] > 0.5:
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
                    if mouse.rotation > 0 and mouse.rotation < 180:
                        sign = -1
                    else:
                        sign = 1
                    mouse.turn(sign*angle)
                    print("no obstacles")
            else:
                selecteds += 1

        draw_window(win, mouses, obstacles)
        for obstacle in obstacles:
            for x,mouse in enumerate(mouses):
                if obstacle.collide(mouse) or mouse.x > 590 or mouse.y > 700:
                    # decreases the fitness score of the neural network
                    # who collided the obstacle
                    ges[x].fitness -= 1
                    mouses.pop(x)
                    networks.pop(x)
                elif mouse.get_rect().top == 0:
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
