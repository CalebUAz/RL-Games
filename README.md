# RL-Games
Getting into Reinforcement Learning

# mlpPolicy:

## Using PPO (Proximal Policy Optimization)

Actor-critic architecture:
1. Policy Network (Actor): Responsible for selecting actions based on the current state.
2. Value Network (Critic): Estimates the value function to provide feedback on how good the state is under the current policy.

## MlpPolicy architecture:
```mermaid
flowchart TD
    subgraph Shared_Layers["Shared Layers"]
        direction TB
        Input["Input Layer"]
        Dense1["Dense Layer 1 (64 neurons, tanh)"]
        Dense2["Dense Layer 2 (64 neurons, tanh)"]
    end

    subgraph Separate_Heads["Separate Heads"]
        direction LR
        PolicyHead["Policy Head (Action Probabilities)"]
        ValueHead["Value Head (State Value)"]
    end

    Input --> Dense1 --> Dense2
    Dense2 --> PolicyHead
    Dense2 --> ValueHead
```

### RL flowchart:
```mermaid
flowchart TD
    A[Start] --> B[Initialize PPO with MlpPolicy]
    B --> C[Define Policy Architecture using policy_kwargs]
    C --> D[Create Environment, e.g., Bowling]
    D --> E[Train the Agent using model.learn]
    E --> F[Save the Trained Model]
    F --> G[Load the Model for Evaluation or Deployment]
    G --> H[Predict Actions and Interact with Environment]
    H --> I[End]

```

## Enviroment setup

Used OpenAI Gym API enviroment to run custom bowling game where:
- `Actions`: The agent controls the ball's velocity in the x and y-directions (continuous action space).
- `Observations`: The state includes the ball's position and the positions of all pins (22-dimensional vector).
- `Reward`: The agent earns rewards proportional to the number of pins knocked down.

This environment provides feedback to the agent by returning a new state, reward, and termination flags after each action.

## Training with PPO

`PPO` is a policy gradient method that trains a stochastic policy to maximize cumulative rewards. It alternates between:
1. Sampling actions from the policy to interact with the environment.
2. Updating the policy using a "clipped surrogate objective" to ensure stable learning while avoiding overly large updates.

## Testing the trained model
1. The trained policy predicts actions based on the current state (obs).
2. The environment executes these actions and returns feedback (new state, reward, etc.).
3. Rendering visually shows how well the agent performs in knocking down pins.