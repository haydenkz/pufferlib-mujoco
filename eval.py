
import torch
import os
import glob
import time

# Import the environment and policy from your training script
from train import make_env, Policy

def run_evaluation(model_path):
    """
    Loads a trained model and evaluates it on the custom CartPole environment.

    Args:
        model_path (str): The path to the saved model checkpoint (.pt).
    """
    # Set the device for PyTorch operations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create the evaluation environment with rendering enabled
    eval_env = make_env(render_mode='human')

    # Instantiate the policy and move it to the correct device
    # The Policy class from your train.py is used here
    policy = Policy(eval_env).to(device)

    # Load the trained model weights from the specified path
    print(f"Loading model from: {model_path}")
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the Policy definition in this script matches the one used for training.")
        eval_env.close()
        return

    # Set the policy to evaluation mode
    policy.eval()

    print("\nStarting evaluation for 1 episode...")

    obs, info = eval_env.reset()
    episode_reward = 0
    done = False
    
    # Set a fixed frame delay for normal speed playback (approx. 50 FPS)
    frame_delay = 1 / 50

    while not done:
        # The environment renders via the viewer.sync() call in the step function
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            # Use the policy's forward pass to get action logits
            logits, _ = policy(obs_tensor)
            # Use argmax for greedy action selection
            action = torch.argmax(logits, dim=-1).item()
            
        next_obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        done = terminated or truncated
        obs = next_obs
        
        # Pause to run at the desired speed
        time.sleep(frame_delay)
            
    print(f"Evaluation finished. Final Reward: {episode_reward}")

    # Clean up the environment
    eval_env.close()

if __name__ == '__main__':
    # Find the latest model file in the experiments directory
    try:
        list_of_files = glob.glob('experiments/*.pt')
        if not list_of_files:
            raise FileNotFoundError("No .pt files found in the 'experiments' directory.")
        latest_model = max(list_of_files, key=os.path.getctime)
    except FileNotFoundError as e:
        print(e)
        exit()

    run_evaluation(model_path=latest_model)
