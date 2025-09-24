import numpy as np
from stable_baselines3 import PPO
import torch
import os
import logging
from importlib import import_module
from model.factory import get_model
import random
import itertools
import torch.nn.functional as F

class V3Model:
    
    def __init__(self, config):
        self.config = config
        self.state_dim = self.config.get("state_dim", 1)
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.build_model()
    
    def build_model(self):
        config = self.config
        method_specs = self.config["method"].split("_")
        self.method_specs = method_specs

        model_config = self.config["model_config"]
        loss_config = self.config["loss_config"]

        if "V3" in method_specs:
            self.model = get_model(config["dataloader"], model_config).to(self.device)
            from model.v3_loss import V3Loss as Loss

        cp_state_dict = torch.load(config["active_checkpoint"])["model"]

        self.model.load_state_dict(cp_state_dict, strict=False)
        self.model.eval()
        self.loss = Loss(loss_config)
        
    def process_data(self, data, action):
        """
        data: torch.Tensor, shape (batch, channel, H, W)
        action: torch.Tensor or np.ndarray, shape (batch, 9), 每个元素为0~1之间的实数
        返回: torch.Tensor, shape (batch, 10, channel, H, W//10)
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(data.device)
        batch, channel, H, W = data.shape
        # 计算切割点
        action = action.clamp(0, 1)
        cut_points = (action * W).long()  # shape: (batch, 9)
        cut_points, _ = torch.sort(cut_points, dim=1)
        # 拼接0和W，得到10个区间
        zeros = torch.zeros((batch, 1), dtype=cut_points.dtype, device=cut_points.device)
        ws = torch.cat([zeros, cut_points, torch.full((batch, 1), W, dtype=cut_points.dtype, device=cut_points.device)], dim=1)  # (batch, 11)
        out = []
        for i in range(10):
            segs = []
            for b in range(batch):
                l, r = ws[b, i].item(), ws[b, i+1].item()
                img = data[b:b+1]  # (1, C, H, W)
                seg = img[..., l:r]  # (1, C, H, r-l)
                # 拉伸到W//10宽度
                seg = F.interpolate(seg, size=(H, W//10), mode='bilinear', align_corners=False)
                segs.append(seg)
            segs = torch.cat(segs, dim=0)  # (batch, C, H, W//10)
            out.append(segs.unsqueeze(1))  # (batch, 1, C, H, W//10)
        out = torch.cat(out, dim=1)  # (batch, 10, C, H, W//10)
        return out.to(self.device)
    
    def step(self, data, action):
        
        processed_data = self.process_data(data, action)
        
        (
            outputs,
            emb_c,
            emb_c_vq,
            vq_indices,
            vq_commit_loss,
            emb_s,
            *rest,
        ) = self.model(processed_data)

        losses = self.loss.compute_loss(
            outputs,
            emb_c,
            emb_c_vq,
            vq_commit_loss,
            emb_s,
            processed_data,
        )
        
        if self.state_dim == 1:
            hidden_state = (data[0], processed_data[0])
        else:
            hidden_state = (data, processed_data)
        
        return hidden_state, losses["total_loss"].item()
    
class SegmentationEnv:

    def __init__(self, config, hidden_dim=4, action_dim=9):
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = V3Model(config)
        self.prepare_data(config)
        self.data = self.get_batch_data()
        self.current_step = 0
        self.max_steps = 1000
        self.portion = 1

    def prepare_data(self, config):
        
        dataloader_module = import_module("dataloader." + config["dataloader"])

        self.S_LIST = dataloader_module.S_LIST
        self.C_LIST = dataloader_module.C_LIST

        self.data_dir = config["data_dir"]
        self.train_loader = dataloader_module.get_dataloader(
            os.path.join(self.data_dir, "train"),
            portion=self.portion,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            n_fragments=config["model_config"]["n_fragments"],
            fragment_len=config["model_config"]["fragment_len"],
            shuffle=True,
        )
        logging.info("Train dataloader ready.")
        self.val_loader = dataloader_module.get_dataloader(
            os.path.join(self.data_dir, "val"),
            portion=self.portion,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            n_fragments=config["model_config"]["n_fragments"],
            fragment_len=config["model_config"]["fragment_len"],
            shuffle=False,
        )
        logging.info("Validation dataloader ready.")
    
    def get_batch_data(self):
        total_batches = len(self.train_loader)
        rand_idx = random.randint(0, total_batches - 1)
        batch = next(itertools.islice(self.train_loader, rand_idx, None))
        return batch
    
    def reset(self):
        self.current_step = 0
        self.data = self.get_batch_data()
        random_actions = np.random.rand(9)
        init_state, _ = self.model.step(self.data, random_actions)
        return init_state

    def step(self, action):
        self.current_step += 1
        hidden_state, loss = self.model.step(action, self.data)
        reward = -loss
        done = self.current_step >= self.max_steps
        return hidden_state, reward, done

class RLTrainer:
    def __init__(self, config):
        self.config = config
        self.env = SegmentationEnv(config)
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log="./ppo_model_env_log"
        )

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def test(self, n_episodes=5):
        #### Visualization ####
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")



