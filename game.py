import pygame
import random

class BowlingGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.ball = pygame.Rect(400, 500, 20, 20)
        self.original_pins = self._initialize_pins()  # Initialize pins in a triangle
        self.pins = self.original_pins.copy()  # Working copy of the pins
        self.score = 0
        self.done = False

    def _initialize_pins(self):
        """Initialize pins in a triangular formation."""
        pins = []
        start_x = 400  # Center of the lane
        start_y = 100  # Starting y-coordinate for the first row
        spacing = 40   # Spacing between pins

        for row in range(4):  # Four rows of pins (1-2-3-4)
            for col in range(row + 1):
                x = start_x + (col - row / 2) * spacing
                y = start_y + row * spacing
                pins.append(pygame.Rect(x - 5, y - 15, 10, 30))  # Pin dimensions: width=10, height=30

        return pins

    def reset(self):
        """Reset the game state."""
        self.ball.center = (400, 500)
        self.pins = self._initialize_pins()  # Reset pins to their original positions
        self.score = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        """Get the current observation of the game state."""
        return {
            "ball": (self.ball.x, self.ball.y),
            "pins": [pin.center for pin in self.pins]
        }

    def step(self, action):
        """Perform an action and update the game state."""
        # Action mapping: 0=left, 1=right, 2=roll
        if action == 0:
            self.ball.x -= 5
        elif action == 1:
            self.ball.x += 5
        elif action == 2:
            self.ball.y -= 5

        reward = 0
        for pin in self.pins[:]:
            if self.ball.colliderect(pin):
                self.pins.remove(pin)
                reward += 10

        if self.ball.y < min(pin.y for pin in self.pins) if self.pins else 100:
            # End game when ball crosses the pin area or all pins are knocked down
            self.done = True
            reward += 50 if not self.pins else -10

        return self._get_obs(), reward, self.done, {}

    def render(self):
        """Render the game state to the screen."""
        self.screen.fill((0, 0, 0))  # Clear screen with black background
        
        # Draw lane
        pygame.draw.rect(self.screen, (200, 200, 200), (100, 50, 600, 500))
        
        # Draw ball
        pygame.draw.circle(self.screen, (255, 0, 0), self.ball.center, int(self.ball.width / 2))
        
        # Draw pins
        for pin in self.pins:
            pygame.draw.rect(self.screen, (255, 255, 255), pin)
        
        pygame.display.flip()

    def draw(self, surface):
        """Draw game elements on a given surface."""
        # Draw lane
        pygame.draw.rect(surface, (200, 200, 200), (100, 50, 600, 500))
        
        # Draw ball
        pygame.draw.circle(surface, (255, 0, 0), self.ball.center, int(self.ball.width / 2))
        
        # Draw pins
        for pin in self.pins:
            pygame.draw.rect(surface, (255, 255, 255), pin)
