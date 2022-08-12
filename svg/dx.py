# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
import torch.nn.functional as F

from . import utils

class SeqDx(nn.Module):
    def __init__(self,
                 env_name,
                 obs_dim, action_dim, action_range,
                 horizon, device,
                 detach_xt,
                 clip_grad_norm,
                 xu_enc_hidden_dim, xu_enc_hidden_depth,
                 x_dec_hidden_dim, x_dec_hidden_depth,
                 rec_type, rec_latent_dim, rec_num_layers,
                 lr,
                 sn):
        super().__init__()

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm

        # Manually freeze the goal locations
        if env_name == 'gym_petsReacher':
            self.freeze_dims = torch.LongTensor([7,8,9])
        elif env_name == 'gym_petsPusher':
            self.freeze_dims = torch.LongTensor([20,21,22])
        else:
            self.freeze_dims = None

        self.rec_type = rec_type
        self.rec_num_layers = rec_num_layers
        self.rec_latent_dim = rec_latent_dim

        self.xu_enc = utils.mlp(
            obs_dim+action_dim, xu_enc_hidden_dim, rec_latent_dim, xu_enc_hidden_depth, sn, sn_first=True, sn_last=True)
        self.x_dec = utils.mlp(
            rec_latent_dim, x_dec_hidden_dim, obs_dim, x_dec_hidden_depth, sn, sn_first=True)

        self.apply(utils.weight_init) # Don't apply this to the recurrent unit.


        mods = [self.xu_enc, self.x_dec]

        if rec_num_layers > 0:
            if rec_type == 'LSTM':
                self.rec = nn.LSTM(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            elif rec_type == 'GRU':
                self.rec = nn.GRU(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            else:
                assert False
            mods.append(self.rec)

        params = utils.get_params(mods)
        self.opt = torch.optim.Adam(params, lr=lr)

    def __getstate__(self):
        d = self.__dict__
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.rec.flatten_parameters()

    def init_hidden_state(self, init_x):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.rec_type == 'LSTM':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == 'GRU':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
        else:
            assert False

        return h

    def unroll_policy(self, state_batch, action_batch, policy, sample=True,
                      last_u=True, detach_xt=False):
        pred_xs = []
        us = []
        log_p_us = []
        """
        xt = init_x
        for t in range(self.horizon-1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:,self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1
        """
#        print(state_batch.shape, 'state-batch')
        state_batch = state_batch.unsqueeze(1)
        action_batch = action_batch.unsqueeze(1)
        n_batch = state_batch.size(1)

        if self.freeze_dims is not None:
            obs_frozen = state_batch[:, :, self.freeze_dims]

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(state_batch)
        xt = state_batch[0].detach()
        for t in range(self.horizon - 1):
            policy_kwargs = {}
            action_batch_i = action_batch[t, :, :].detach()
            next_s_batch_i = state_batch[t + 1, :, :].detach()
            action_mean, _, _, action_std = policy(xt, **policy_kwargs, return_var=True)
        #    action_mean, _, _, _ = policy(xt, **policy_kwargs, return_var=True)
        #    action_std = 5e-3
            with torch.no_grad():
                varsigma = (torch.arctanh(action_batch_i) - action_mean) / action_std
           #     print(varsigma, action_batch_i, action_mean)
            re_action = torch.tanh(action_mean + varsigma * action_std)
            #     _, _, log_p_ut = policy(xt, **policy_kwargs)
            #    _, re_action, log_p_ut = policy(xt, **policy_kwargs)
            us.append(re_action)
            re_policy = utils.SquashedNormal(action_mean, action_std)
            log_p_ut = re_policy.log_prob(re_action).sum(-1, keepdim=True)
            log_p_us.append(log_p_ut)

            xut = torch.cat((xt, re_action), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            model_std = 5e-3
            with torch.no_grad():
                xi = (next_s_batch_i - xtp1) / model_std
            re_next_s = xtp1 + xi * model_std
            pred_xs.append(re_next_s)
            xt = re_next_s

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim)
        else:
            pred_xs = torch.stack(pred_xs)
        return us, log_p_us, pred_xs


    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(x)

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
          #  xtp1 = self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:,self.freeze_dims] = obs_frozen
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs


    def forward(self, x, us):
        return self.unroll(x, us)


    def update_step(self, obs, action, reward, logger, step, train_model=True):
        if not train_model:
            logger.log('train_model/obs_loss', 0, step)
            return
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        pred_obs = self.unroll(obs[0], action[:-1], detach_xt=self.detach_xt)
        target_obs = obs[1:]
        assert pred_obs.size() == target_obs.size()

        obs_loss = F.mse_loss(pred_obs, target_obs, reduction='mean')

        self.opt.zero_grad()
        obs_loss.backward()
        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]['params']
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.opt.step()

        logger.log('train_model/obs_loss', obs_loss, step)

        return obs_loss.item()
