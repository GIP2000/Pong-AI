import pygame
from pygame.locals import *
import os
import math
import neat
SIZE = SCREEN_WIDTH,SCREEN_HEIGHT = 1000,1000


class Paddle: 
    WIDTH = 10
    HEIGHT = 50
    IMG = pygame.image.load(os.path.join("resources/images","Paddle.png"))

    def __init__(self,is_player_1):
        self.x = 50 if is_player_1 else SCREEN_WIDTH-50
        self.y = round(SCREEN_HEIGHT/2)
        
        

    def up(self):
        self.y-= 10 if self.y - 10 >= 0 else 0 
    
    def down(self):
        self.y += 10 if self.y + 10 + Paddle.HEIGHT<= SCREEN_HEIGHT else 0 
    
    def get_mask(self):
        return pygame.mask.from_surface(Paddle.IMG)

    def draw(self,win):
        win.blit(Paddle.IMG,(self.x,self.y))
    

    


class Ball:
    RADIUS = 10
    IMG =  pygame.image.load(os.path.join("resources/images","Ball.png"))
    VEL = 10

    def __init__(self):
        self.x = round(SCREEN_WIDTH/2)
        self.y  = round(SCREEN_HEIGHT/2)
        self.angle = 0

    def draw(self,win):
        win.blit(Ball.IMG,(self.x,self.y))

    def move(self):
        self.x += round(Ball.VEL*math.cos(self.angle))

        if self.x <= 0:
            return 1
        elif self.x >= SCREEN_WIDTH:
            return 2

        if self.y + round(Ball.VEL*math.sin(self.angle)) <=0 or self.y + round(Ball.VEL*math.sin(self.angle)) >= SCREEN_HEIGHT: 
            self.angle += (math.pi/2) 

        self.y += round(Ball.VEL*math.sin(self.angle)) 

        return 0

    def check_collision(self,player1,player2):

        def find_collision_point(self,player):
            ball_mask = pygame.mask.from_surface(Ball.IMG)
            player_mask = player.get_mask()
            offset = (self.x - player.x, self.y - player.y)

            c_point = player_mask.overlap(ball_mask,offset)
            status = bool(c_point)

            if status:
                self.angle = (self.y - Paddle.HEIGHT/2)/(Paddle.HEIGHT/2)*math.pi/4
                return True
            else:
                return False

        if self.x <= 60:
            return find_collision_point(self,player1),0
        elif self.x >= SCREEN_WIDTH-60:
            return find_collision_point(self,player2),1 
        else:
            return False,None



def main_game(genomes,config):
    
    pygame.init()
    WIN = pygame.display.set_mode(SIZE)
    clock = pygame.time.Clock()
    running = True
    player1,player2 = Paddle(True), Paddle(False)
    ball = Ball()

    neat1 = neat.nn.FeedForwardNetwork.create(genomes[0][1],config)
    neat2 = neat.nn.FeedForwardNetwork.create(genomes[1][1],config)
    genomes[0][1].fitness = 0
    genomes[1][1].fitness = 0
    
    def activate(player1,ball,nn):
        output = nn.activate((player1.y,player1.x,ball.x,ball.y))[0]
        if output >= 0 and output <= 1/3:
            player1.up()
        elif output >= 1/4 and output <= 2/3:
            player1.down()


    while running:
        clock.tick(30)
                     
        activate(player1,ball,neat1)
        activate(player2,ball,neat2)

        WIN.fill((0,0,0))
        lose = ball.move()
        if lose == 1:
            genomes[0][1] = None # genomes[0][1].fitness -=50
            running = False
        elif lose == 2:
            genomes[1][1] = None # genomes[1][1].fitness -=50
            running = False
        
        collide,who_collide = ball.check_collision(player1,player2)
        if collide:
            print(who_collide)
            genomes[who_collide][1].fitness +=1


        player1.draw(WIN)
        player2.draw(WIN)
        ball.draw(WIN)
        
        
        pygame.display.flip()



def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)   
    
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

     # Run for up to 50 generations.
    winner = p.run(main_game, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    run(config_path)