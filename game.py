import pygame
import random

class BowlingGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.ball = pygame.Rect(400, 500, 20, 20)
        # Initialize original pins once
        self.original_pins = [pygame.Rect(100 + i*50, 100, 10, 30) for i in range(10)]
        self.pins = self.original_pins.copy()  # Working copy
        self.score = 0
        self.done = False

    def reset(self):
        self.ball.center = (400, 500)
        # Reset to original pins
        self.pins = self.original_pins.copy()
        self.score = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return {
            "ball": (self.ball.x, self.ball.y),
            "pins": [pin.center for pin in self.pins]
        }

    def step(self, action):
        # Action mapping: 0=left, 1=right, 2=roll
        if action == 0: self.ball.x -= 5
        elif action == 1: self.ball.x += 5
        elif action == 2: self.ball.y -= 5

        reward = 0
        for pin in self.pins[:]:
            if self.ball.colliderect(pin):
                self.pins.remove(pin)
                reward += 10

        if self.ball.y < 100: 
            self.done = True
            reward += 50 if not self.pins else -10

        return self._get_obs(), reward, self.done, {}

    def render(self):
        self.screen.fill((0,0,0))
        pygame.draw.rect(self.screen, (255,255,255), self.ball)
        for pin in self.pins:
            pygame.draw.rect(self.screen, (255,0,0), pin)
        pygame.display.flip()
        
    def draw(self, surface):
        # Draw lane
        pygame.draw.rect(surface, (200, 200, 200), (0, 300, 800, 300))
        
        # Draw ball
        pygame.draw.circle(surface, (255, 0, 0), self.ball.center, 10)
        
        # Draw pins
        for pin in self.pins:
            pygame.draw.rect(surface, (255, 255, 255), pin)
