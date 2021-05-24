import gym
import collections
from tensorboardX import SummaryWriter

GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        self.state = self.env.reset()
        
        # defaultdict is much like an ordinary dict, except that when you try to\
        # access or modify a key that’s not present in the dict, a default value\
        # is automatically given to that key (contrary to dict)(missing key error\
        # handling)
        
        self.values = collections.defaultdict(float)
  
        
    def play_sample_trajectory(self):       
        act = self.env.action_space.sample()
        state = self.state
        next_state, reward, is_done, _ = self.env.step(act)
        self.state = self.env.reset() if is_done else next_state
        return (state, act, reward, next_state)
            
    
    def q_update(self, state, act, reward, next_state):
        best_next_val, _ = self.best_action_state_value(next_state)                          
        self.values[(state, act)] = (1-ALPHA)*self.values[(state, act)] + \
            ALPHA*(reward + GAMMA*best_next_val)
                
            
    def best_action_state_value(self, state):
        selected_act, best_val = None, None
        for act in range(self.env.action_space.n):
            val = self.values[(state, act)]
            if best_val is None or best_val < val:
                best_val = val
                selected_act = act
        return best_val, selected_act
        
    
    def play_game(self, env):
        state = env.reset()
        total_reward = 0.0
        while True:
            _, act = self.best_action_state_value(state)
            next_state, reward, is_done, _ = env.step(act)
            
            #Here, we do not learn, but only play! That is a reason why it takes
            #more iterations to solve the game!
           
            total_reward += reward
            
            if is_done:
                break
            state = next_state
            
        return total_reward    
            
            

if __name__ == "__main__":
    test_env = gym.make("FrozenLake-v0")
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")
    
    iteration = 0
    best_reward = 0.0
    
    while True:
        iteration += 1
        s, a, r, s_pr = agent.play_sample_trajectory()
        agent.q_update(s, a, r, s_pr)
        
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_game(test_env)
            
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iteration)
        if reward > best_reward:
            best_reward = reward
            print("Best reward in iteration %d: %.3f" %(iteration, best_reward))
            
        if reward > 0.80:
            print("Solved!")
            break       
        
    writer.close()
    
    
    