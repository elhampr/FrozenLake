import gym
import collections
from tensorboardX import SummaryWriter

GAMMA = 0.9
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        self.state = self.env.reset()
        
        # defaultdict is much like an ordinary dict, except that when you try to\
        # access or modify a key that’s not present in the dict, a default value\
        # is automatically given to that key (contrary to dict)(missing key error\
        # handling)
        
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
        
    def play_sample_episodes(self, episodes):
        for _ in range(episodes):
            act = self.env.action_space.sample()
            next_state, reward, is_done, _ = self.env.step(act)
            self.rewards[(self.state, act, next_state)] = reward
            
            #Below, we define the dictionary with different key vs. above,
            #because, later we want to do sum over the new state values.
            
            self.transitions[(self.state, act)][next_state] += 1
            self.state = self.env.reset() if is_done else next_state
            
    
    def action_state_value(self, act, state):
        poss_next_states = self.transitions[(state,act)]
        total = sum(poss_next_states.values())
        act_val = 0.0
        for next_state, count in poss_next_states.items():
            reward = self.rewards[(state,act,next_state)]
            act_val += (count/total)*(reward + GAMMA*self.values[next_state])
        return act_val
        
    
    def value_iteration(self):
        for state in range(self.env.observation_space.n): 
            act_vals = [self.action_state_value(act,state) \
                        for act in range(self.env.action_space.n)]
            self.values[state] = max(act_vals)        
                
   
    def select_action(self, state):
        selected_act, best_val = None, None
        for act in range(self.env.action_space.n):
            val = self.action_state_value(act, state)
            if best_val is None or best_val < val:
                best_val = val
                selected_act = act
        return selected_act
        
    
    def play_game(self, env):
        state = env.reset()
        total_reward = 0.0
        while True:
            act = self.select_action(state)
            next_state, reward, is_done, _ = env.step(act)
            
            #Here, we also learn by updating rewards and transition matrices, to\
            #take advantage from all the generated information including the test\
            #phase (which is here)
           
            self.rewards[(state,act,next_state)] = reward
            self.transitions[(state, act)][next_state] += 1
            
            total_reward += reward
            
            if is_done:
                break
            state = next_state
            
        return total_reward    
            
            

if __name__ == "__main__":
    test_env = gym.make("FrozenLake-v0")
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")
    
    iteration = 0
    best_reward = 0.0
    
    while True:
        iteration += 1
        agent.play_sample_episodes(100)
        agent.value_iteration()
        
        reward = 0
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
    
    
    