import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_cumulative_reward(csv_file):
    # Load data
    
    df = pd.read_csv(csv_file)
    
    # Compute cumulative rewards per episode
    df['Computed_Cumulative_Rewards'] = df.groupby('Episode')['Rewards'].cumsum()
    
    # Get the last timestep of each episode for plotting
    final_rewards = df.groupby('Episode').last()
    
    # Plot cumulative rewards
    plt.figure(figsize=(10, 5))
    plt.plot(final_rewards.index, final_rewards['Computed_Cumulative_Rewards'], marker='o', linestyle='-')
    
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Episode")
    plt.grid()
    plt.show()

def plot_timestep_reward(csv_file, episode_num):
    # Load data
    
    df = pd.read_csv(csv_file)
    
    # Compute cumulative rewards per episode
    df_episode = df[df['Episode'] == episode_num]
    
    
    # Plot cumulative rewards
    plt.figure(figsize=(10, 5))
    plt.plot(df_episode['Timestep'], df_episode['Rewards'],  marker='o', linestyle='-')
    
    plt.xlabel("Timestep")
    plt.ylabel("Rewards")
    plt.title(f"reward per timestep for episode {episode_num}")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    file_names = ['/home/liza/SnakeRobot/CoadaptationCode/2025_03_14-17_31_56Rewards_Design3']  # Replace with your actual file names
    for i in range(50):
        for file_path in file_names:
            counter = 4
            df = pd.read_csv(file_path, delimiter='\t')  # Change delimiter if needed
            df.to_csv(f'output_{counter}.csv', index=False)

        #put all csv files in a list
            designs = ['output_4.csv']
        
            with open(designs[0], 'r') as file :
                filedata = file.read()

                filedata = filedata.replace('"', '')
                with open(designs[0], 'w') as file:
                    file.write(filedata) 
            plot_timestep_reward(f'output_4.csv', episode_num=i)
            plot_cumulative_reward('output_4.csv')
