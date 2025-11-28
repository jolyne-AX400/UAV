
import torch
from env import SimpleEnv
from models import ActorGAT, CriticCentral
from graph_utils import build_adj_matrix
from train import node_features_from_obs, flatten_global

def evaluate(actor_path, episodes=5, comm_range=10.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = SimpleEnv(seed_value=0)
    ckpt = torch.load(actor_path, map_location=device)
    node_in_dim = ckpt.get('node_feat_dim', 8)
    actor = ActorGAT(node_in_dim=node_in_dim).to(device)
    actor.load_state_dict(ckpt['actor_state_dict'])
    actor.eval()
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            feats = node_features_from_obs(obs, max_energy=env.sg.V.get('ENERGY', 3000.0)).to(device)
            positions = torch.tensor([o['uav_position'] for o in obs], dtype=torch.float32).to(device)
            adj = build_adj_matrix(positions, comm_range=comm_range).to(device)
            with torch.no_grad():
                mean, std = actor(feats, adj)
                actions = mean  # deterministic
            next_obs, rewards, done, info = env.step(actions.cpu().numpy().tolist())
            total_reward += sum(rewards)
            obs = next_obs
        print(f"Eval EP {ep} | total_reward {total_reward:.2f} | explored_ratio {info.get('explored_ratio',0.0):.4f}")

if __name__ == '__main__':
    evaluate('checkpoints/final.pt', episodes=3)
