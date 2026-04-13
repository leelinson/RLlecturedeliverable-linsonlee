import gymnasium as gym
import numpy as np

num_episodes      = 10_000  # how many games to play during training
learning_rate     = 0.1     # how fast the agent updates what it learns (alpha)
discount          = 0.99    # how much the agent cares about future rewards (gamma)
exploration_rate  = 1.0     # how often the agent tries random actions (epsilon)
exploration_decay = 0.995   # how quickly the agent stops exploring over time
min_exploration   = 0.01    # always keep a tiny bit of exploration
num_buckets       = 20      # how finely we divide up each observation into slots

obs_min = np.array([-2.4, -4.0, -0.2095, -4.0])
obs_max = np.array([ 2.4,  4.0,  0.2095,  4.0])

def get_discrete_state(observation):
    clipped     = np.clip(observation, obs_min, obs_max)
    normalized  = (clipped - obs_min) / (obs_max - obs_min)  
    bucket_ids  = (normalized * num_buckets).astype(int)
    return tuple(np.clip(bucket_ids, 0, num_buckets - 1))

# A big lookup table: for every possible state, store how good each action is.
# Starts at zero — the agent knows nothing and learns from scratch.
# Shape: (20, 20, 20, 20, 2) — 20 buckets per observation, 2 possible actions
q_table = np.zeros([num_buckets] * 4 + [2])

env            = gym.make("CartPole-v1")
rewards_per_episode = []

for episode in range(num_episodes):
    raw_observation, _ = env.reset()
    current_state      = get_discrete_state(raw_observation)
    total_reward       = 0
    game_over          = False

    while not game_over:
        # Decide whether to explore (random) or exploit (use what we've learned)
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()           
        else:
            action = np.argmax(q_table[current_state])   

        # Take the action and see what happens
        new_observation, reward, terminated, truncated, _ = env.step(action)
        game_over  = terminated or truncated
        next_state = get_discrete_state(new_observation)

        # Update the Q-table using what we just experienced:
        # "What I thought this action was worth" vs "What it actually led to"
        best_future_value  = np.max(q_table[next_state])
        current_value      = q_table[current_state][action]
        what_it_was_worth  = reward + discount * best_future_value * (not game_over)

        q_table[current_state][action] += learning_rate * (what_it_was_worth - current_value)

        current_state = next_state
        total_reward += reward

    # Reduce exploration over time — trust what we've learned more and more
    exploration_rate = max(min_exploration, exploration_rate * exploration_decay)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 1000 == 0:
        recent_avg = np.mean(rewards_per_episode[-1000:])
        print(f"Episode {episode+1:>6} | Avg score (last 1000): {recent_avg:.1f} | Exploration: {exploration_rate:.3f}")

env.close()

# Now test the learned policy with no exploration — pure exploitation
print("\nTesting the learned policy over 100 games...")
test_env     = gym.make("CartPole-v1")
test_scores  = []

for _ in range(100):
    raw_observation, _ = test_env.reset()
    current_state      = get_discrete_state(raw_observation)
    score = 0
    game_over = False

    while not game_over:
        action                             = np.argmax(q_table[current_state])
        raw_observation, reward, terminated, truncated, _ = test_env.step(action)
        game_over     = terminated or truncated
        current_state = get_discrete_state(raw_observation)
        score        += reward

    test_scores.append(score)

test_env.close()
print(f"  Average score : {np.mean(test_scores):.1f}")
print(f"  Best score    : {np.max(test_scores):.1f}")
print(f"  Solved (≥475) : {'Yes ✓' if np.mean(test_scores) >= 475 else 'Not yet — try more episodes'}")