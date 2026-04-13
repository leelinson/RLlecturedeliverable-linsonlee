import gymnasium as gym
import numpy as np

# ─── Hyperparameters (matching lecture notation) ───────────────────────────────
EPISODES      = 10_000
ALPHA         = 0.1      # learning rate
GAMMA         = 0.99     # discount factor
EPSILON       = 1.0      # starting exploration rate (ε-greedy)
EPSILON_DECAY = 0.995
EPSILON_MIN   = 0.01
BINS          = 20       # discretization buckets per state dimension

# ─── Discretization ────────────────────────────────────────────────────────────
# CartPole gives continuous states, but tabular Q-learning needs discrete states.
# We clip each observation to a known range and bin it into BINS buckets.
# State: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
OBS_LOW  = np.array([-2.4, -4.0, -0.2095, -4.0])
OBS_HIGH = np.array([ 2.4,  4.0,  0.2095,  4.0])

def discretize(obs):
    """Map continuous observation s_t into a discrete tuple (index into Q-table)."""
    obs_clipped = np.clip(obs, OBS_LOW, OBS_HIGH)
    ratios      = (obs_clipped - OBS_LOW) / (OBS_HIGH - OBS_LOW)  # normalize to [0, 1]
    indices     = (ratios * BINS).astype(int)
    return tuple(np.clip(indices, 0, BINS - 1))

# ─── Q-Table: Q(s, a) for all s ∈ S, a ∈ A ────────────────────────────────────
# Initialize Q(s, a) = 0 for all s, a  (Algorithm 1, line 1)
q_table = np.zeros([BINS] * 4 + [2])

# ─── Algorithm 1: Tabular Q-Learning (from lecture notes) ─────────────────────
env     = gym.make("CartPole-v1")
epsilon = EPSILON
episode_rewards = []

for episode in range(EPISODES):                          # for each episode (line 2)
    obs, _       = env.reset()
    s_t          = discretize(obs)                       # initialize state s_0 (line 3)
    total_reward = 0
    done         = False

    while not done:                                      # for each step t (line 4)

        # ε-greedy action selection (line 5)
        # with prob ε → random action (explore)
        # with prob 1-ε → argmax_a Q(s_t, a) (exploit)
        if np.random.random() < epsilon:
            a_t = env.action_space.sample()
        else:
            a_t = np.argmax(q_table[s_t])

        # Take action a_t, observe r_t and s_{t+1} (line 6)
        next_obs, r_t, terminated, truncated, _ = env.step(a_t)
        done      = terminated or truncated
        s_t_next  = discretize(next_obs)

        # Q-learning update rule (line 7):
        # Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ * max_a' Q(s_{t+1}, a') - Q(s_t, a_t) ]
        #                                  |_________TD target_________|   |__current estimate__|
        td_target = r_t + GAMMA * np.max(q_table[s_t_next]) * (not done)
        td_error  = td_target - q_table[s_t][a_t]
        q_table[s_t][a_t] += ALPHA * td_error

        s_t = s_t_next                                   # s_t ← s_{t+1} (line 8)
        total_reward += r_t

    # Decay ε after each episode (explore early, exploit later)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    if (episode + 1) % 1000 == 0:
        avg = np.mean(episode_rewards[-1000:])
        print(f"Episode {episode+1:>6} | Avg reward (last 1000): {avg:.1f} | ε: {epsilon:.3f}")

env.close()

# ─── Evaluation: greedy policy, no exploration ─────────────────────────────────
print("\nEvaluating learned policy (greedy, ε=0) over 100 episodes...")
eval_env     = gym.make("CartPole-v1")
eval_rewards = []

for _ in range(100):
    obs, _ = eval_env.reset()
    s_t    = discretize(obs)
    total  = 0
    done   = False
    while not done:
        a_t                    = np.argmax(q_table[s_t])   # always exploit
        obs, r, term, trunc, _ = eval_env.step(a_t)
        done  = term or trunc
        s_t   = discretize(obs)
        total += r
    eval_rewards.append(total)

eval_env.close()

print(f"  Mean reward : {np.mean(eval_rewards):.1f}")
print(f"  Max reward  : {np.max(eval_rewards):.1f}")
print(f"  Solved (≥475): {'Yes ✓' if np.mean(eval_rewards) >= 475 else 'Not yet — try more episodes'}")