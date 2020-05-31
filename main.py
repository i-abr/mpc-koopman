import numpy as np
import random
import pickle
from datetime import datetime

import sys
import os
sys.path.append('../')

# local imports
import envs

import torch
from mpc_lib import ShootingMethod
from mpc_lib import MPPI

from koopman_linear_embedding import ModelOptimizer, LinEmbModel, ReplayBuffer
from normalized_actions import NormalizedActions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-4)
parser.add_argument('--policy_lr',  type=float, default=3e-4)

parser.add_argument('--seed', type=int, default=666)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=2)

parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()


if __name__ == '__main__':


    env_name = args.env
    try:
        env = NormalizedActions(envs.env_list[env_name](render=args.render))
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = NormalizedActions(envs.env_list[env_name]())
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'
    env.reset()

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)


    # now = datetime.now()
    # date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
    # path = './data/' + env_name +  '/' + 'h_sac/' + date_str
    # if os.path.exists(path) is False:
    #     os.makedirs(path)


    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]

    model = LinEmbModel(state_dim, action_dim, z_dim=state_dim+32).to(device)

    replay_buffer_size = 100000
    replay_buffer = ReplayBuffer(replay_buffer_size)
    optimizer = ModelOptimizer(model, replay_buffer, lr=args.model_lr)

    # mpc_planner = MPPI(model, T=args.horizon, eps=0.3)
    mpc_planner = ShootingMethod(model, T=args.horizon)
    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        mpc_planner.reset()

        episode_reward = 0
        done = False
        for step in range(max_steps):

            if frame_idx > 200:
                action = mpc_planner.update(state)
                eps = 1.0 * (0.995**(frame_idx-200))
                action = action + np.random.normal(0., eps, size=(action_dim,))
            else:
                action = env.action_space.sample()

            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            replay_buffer.push(state, action, reward, next_state)

            if len(replay_buffer) > batch_size:
                optimizer.update_model(batch_size, epochs=args.model_iter)
                print(optimizer.log['loss'][-1])

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % (max_frames//10) == 0:
                last_reward = rewards[-1][1] if len(rewards)>0 else 0
                print(
                    'frame : {}/{}, \t last rew: {}'.format(
                        frame_idx, max_frames, last_reward
                    )
                )

                # pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                # torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                # torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break

        print('ep rew', ep_num, episode_reward)
        rewards.append([frame_idx, episode_reward])
        ep_num += 1
    print('saving final data set')
    # pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    # torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
    # torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
