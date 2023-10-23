import pygame
import neat
import os
import math

pygame.font.init()
WIN_WIDTH = 600
WIN_HEIGHT = 800
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

obstacle = pygame.image.load(os.path.join("images","obstacle.png")).convert_alpha()
mouse = pygame.image.load(os.path.join("images","mouse.png")).convert_alpha()
background = pygame.transform.scale2x(pygame.image.load(os.path.join("images","background.png")))

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


def draw_window(win, mouse, obstacles):
    win.blit(background, (0,0))
    mouse.draw(win)
    for obstacle in obstacles:
        obstacle.draw(win)
    pygame.display.update()



def main():
    networks = []
    genomes = []
    mouse = Mouse(100,700)
    for genome in genomes:
        network = neat.nn.FeedForwardNetwork(genome,config)
        networks.append(network)
    obstacle1 = Obstacle(150,600)
    #obstacle2 = Obstacle(400,200)
    obstacles = [obstacle1]
    collision = False
    run = True
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    collide = False
    while run:
        print(mouse.rotation)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        # if not collide:
        mouse.turn(0.1)
        mouse.move(0.1)
        draw_window(win, mouse, obstacles)
        for obstacle in obstacles:
            if obstacle.collide(mouse) or mouse.get_rect().left <0 or mouse.get_rect().top > 800:
                # decreases the fitness score of the neural network
                # who collided the obstacle
                collide = True
    pygame.quit()
    quit()
main()
"""
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main(),50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-freeforward.txt")
    run(config_path)
"""
