import pygame
from pygame.locals import *
import os
import math
import random 
from QLeanring import QLearningHandler as QL

SIZE = SCREEN_WIDTH,SCREEN_HEIGHT = 1000,1000

global_reward1 = 0
global_reward2 = 0

class Paddle: 
    WIDTH = 10
    HEIGHT = 50
    IMG = pygame.image.load(os.path.join("resources/images","Paddle.png"))

    def __init__(self,is_player_1):
        self.x = 50 if is_player_1 else SCREEN_WIDTH-50
        self.y = round(SCREEN_HEIGHT/2)
        
    def action(self,action):
        if action == 1:
            self.up()
        elif action == 2:
            self.down()

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
        self.angle = 0 if random.randint(1,2) == 1 else math.pi


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
                if self.x <=60:
                    self.angle += math.pi 
                return True
            else:
                return False

        if self.x <= 60:
            return find_collision_point(self,player1),0
        elif self.x >= SCREEN_WIDTH-60:
            return find_collision_point(self,player2),1 
        else:
            return False,None



def main_game(Q1,Q2,train=False,episode=0,show_often=1):
    
    render = not train or episode%show_often == 0
    global global_reward1
    global global_reward2  # added as a flag to see if it will shed insight 

    if render:
        pygame.init()
        WIN = pygame.display.set_mode(SIZE)
        clock = pygame.time.Clock()
    running = True
    player1,player2 = Paddle(True), Paddle(False)
    ball = Ball()

    state1 = [ball.x,ball.y,player1.y]
    state2 = [ball.x,ball.y,player2.y]

    while running:
        if render:
            clock.tick(30)

        action1 = Q1.choose_action(state1)
        action2 = Q2.choose_action(state2)
        player1.action(action1)
        player2.action(action2)

        new_state1 = [ball.x,ball.y,player1.y] 
        new_state2 = [ball.x,ball.y,player2.y]     
       
        collide,who_collide = ball.check_collision(player1,player2)
        
        if collide:
            if who_collide == 0:
                reward1 = 1
                reward2 = 0
            else:
                reward2 = 1
                reward1 = 0
        else:
            reward2 = 0
            reward1 = 0

        Q1.update_q(state1,new_state1,action1,reward1)
        Q2.update_q(state2,new_state2,action2,reward2)

        global_reward1+=reward1
        global_reward2+=reward2

        
        lose = ball.move()
        if lose == 1:
            Q2.winning_move(state2,action2)  # idk if this is actually good for my use case 
            running = False
        elif lose == 2:
            Q1.winning_move(state1,action1)
            running = False
        
        if render:
            WIN.fill((0,0,0))
            player1.draw(WIN)
            player2.draw(WIN)
            ball.draw(WIN)
        
        if render:
            pygame.display.flip()

    return Q1,Q2
    

    

if __name__ == "__main__":
    episodes = 100000
    step = 1000
    Q1 = QL(3,[1000,1000,1000],[0,0,0],[10,10,10],episodes=episodes,use_epsilon=True)
    Q2 = QL(3,[1000,1000,1000],[0,0,0],[10,10,10],episodes=episodes,use_epsilon=True)
    for i in range(episodes):
        Q1,Q2 = main_game(Q1,Q2,True,i,step)
        if i%step == 0:
            print("Gen {} reward1 = {} reward2 = {}".format(i,global_reward1,global_reward2))
        Q1.epsilon_decay()
        Q2.epsilon_decay()

    

    
