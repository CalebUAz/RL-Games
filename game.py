import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class BowlingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(BowlingEnv, self).__init__()
        self.width, self.height = 800, 600
        self.screen = None  # For rendering
        self.clock = pygame.time.Clock()

        # Define action and observation spaces
        # Actions: [x_velocity, y_velocity] normalized between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations: Ball position + pin positions (22 dimensions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(self.width, self.height),
            shape=(22,),
            dtype=np.float32,
        )

        # Game elements
        self.ball = pygame.Rect(400, 500, 20, 20)
        self.pins = [pygame.Rect(100 + i * 50, 100, 20, 40) for i in range(10)]
        self.score = 0

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        # Set the random seed for reproducibility
        super().reset(seed=seed)
        np.random.seed(seed)

        # Reset game elements
        self.ball.center = (400, 500)
        self.pins = [pygame.Rect(100 + i * 50, 100, 20, 40) for i in range(10)]
        self.score = 0

        return self._get_state(), {}

    def step(self, action):
        """Applies an action and returns the next state, reward, terminated flag, truncated flag, and info."""
        # Scale action to velocity
        ball_velocity = [action[0] * 10, action[1] * 10]
        
        # Update ball position
        self.ball.x += ball_velocity[0]
        self.ball.y += ball_velocity[1]

        # Collision detection
        remaining_pins = []
        for pin in self.pins:
            if not self.ball.colliderect(pin):
                remaining_pins.append(pin)
            else:
                self.score += 1

        self.pins = remaining_pins
        reward = self.score * 0.1

        # Termination conditions
        terminated = False
        if len(self.pins) == 0:  # All pins knocked down
            terminated = True

        # Truncation condition (e.g., ball goes out of bounds)
        truncated = False
        if self.ball.y < 50:  # Ball reaches end of lane or leaves bounds
            truncated = True

        # Return state, reward, termination flags, and info
        return self._get_state(), reward, terminated, truncated, {}


    def render(self):
        """Renders the environment."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.screen.fill((0, 0, 0))
        
        # Draw lane
        pygame.draw.rect(self.screen, (200, 200, 200), (0, 300, 800, 300))
        
        # Draw ball
        pygame.draw.circle(self.screen, (255, 0, 0), self.ball.center, 10)
        
        # Draw pins
        for pin in self.pins:
            pygame.draw.rect(self.screen, (255, 255, 255), pin)
            
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        """Closes the environment."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _get_state(self):
        """Returns the current state as a numpy array with a fixed size of 22."""
        # Ball position
        state = [self.ball.x, self.ball.y]
        
        # Pin positions (pad with -1 for missing pins)
        for pin in self.pins:
            state.append(pin.x)
            state.append(pin.y)
        
        # Pad with -1 if fewer than 10 pins remain
        while len(state) < 22:
            state.append(-1)
        
        # Ensure the state has exactly 22 elements
        return np.array(state[:22], dtype=np.float32)


