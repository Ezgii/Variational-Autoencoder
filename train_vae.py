import random
import torch
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch import optim

from MyVAE import MyVAE

debug = False # Set to True for debugging the code
DEF_Q3 = False # Set to True to reset the environment after 20 time steps, for Question 3
DEF_Q4 = False # Set to True to collect images with certain pole angle, for Question 4
DEF_Q5 = False # Set to True to collect the images from Mountain Car environment, for Question 5

#print(gym.__version__)

# we will crop the image to remove the top and bottom (those are always white)
crop_proportions = (0.4, 0.0, 1.0, 1.0)

# after the crop, we will reduce the image size to these dimensions for faster training
img_dim = (64, 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vae():

    # initialize the gym environment
    #########################

    # try different environments
    if DEF_Q5 == False:
        env = gym.make("CartPole-v1", render_mode='rgb_array') 
    elif DEF_Q5 == True:
        env = gym.make("MountainCar-v0", render_mode='rgb_array')
        #env = gym.make("Pendulum-v1", render_mode='rgb_array')

    #########################
    # first observation from the environment
    obs = env.reset()
    #img = env.render(mode='rgb_array')
    img = env.render()
    crop_dim = (
        int(crop_proportions[0] * img.shape[0]),
        int(crop_proportions[1] * img.shape[1]),
        int(crop_proportions[2] * img.shape[0]),
        int(crop_proportions[3] * img.shape[1])
    )

    # VAE

    input_channels = 3
    latent_dim = 10
    training_size = 2000
    batch_size = latent_dim * 10
    n_epochs = 400

    # initialize the VAE
    # VAE model
    vae = MyVAE(
        in_channels=input_channels,
        latent_dim=latent_dim,
    ).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    imgs = np.zeros((training_size, input_channels, *img_dim), dtype=np.float32)

    # Collect pixel data from the gym

    # episode frame counter
    frame_idx = 0
    
    # environment reset counter
    env_rst_cntr = 0
    
    i = 0
    #for i in range(training_size):
    while i < training_size:
    
        frame_idx += 1

        # get a random action in this environment
        action = env.action_space.sample()

        # obs is observation data from the env. 
        # Look at the gym code to find which one is a pole angle. 
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        
        ### obs, reward, done, info = env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # get pixel observations, crop, and resize
        # img = env.render(mode='rgb_array')
        img = env.render()
        
        #if debug:
            # render in human mode
            #plt.imshow(img)
            #plt.pause(0.01)o
            
        img = img[crop_dim[0]: crop_dim[2], crop_dim[1]: crop_dim[3], :]
        img = cv2.resize(img, dsize=img_dim, interpolation=cv2.INTER_CUBIC)
       
        if debug:
            # how the model will see the image after crop and resize
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            
        img = img.swapaxes(0, 2).reshape((1, input_channels, *img_dim)).astype(np.float32) / 255.0

        #################
        
        if DEF_Q4 == False: # original code
            imgs[i] = img
            i += 1
            
        elif DEF_Q4 == True:  # add some conditional logic to save the images you need
            if obs[2] > -0.025 and obs[2] < 0.025 : # collect data if the pole angle is in this range
                imgs[i] = img
                i += 1

        #################

        #################

        if DEF_Q3 == False: # original code
            if done:
                obs = env.reset()
                frame_idx = 0
                
        elif DEF_Q3 == True: # update the reset conditions to save the images you need
            env_rst_cntr += 1
            if done or (env_rst_cntr == 20):
                obs = env.reset()
                frame_idx = 0
                env_rst_cntr = 0

        #################

    env.close()

    # visualization init
    plt.ion()
    plt.show()

    # train VAE
    for i in range(n_epochs):
        # observations for cvae to use as labels
        start_idx = random.randint(0, training_size - batch_size)

        train_imgs = imgs[start_idx : start_idx + batch_size]

        out_imgs = vae(
            torch.from_numpy(train_imgs.copy()).to(device),
        )
        loss = vae.loss(*out_imgs, kl_w=0.0005)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)

        # get a few generated images
        rand_idx = np.random.randint(0, batch_size - 1)
        im = out_imgs[0][rand_idx: rand_idx + 1].detach().cpu().numpy().reshape(
            (1, 3, *img_dim)).swapaxes(1, 3)
        im = (im * 255.0).astype(np.uint8)

        # show generated image
        plt.subplot(
            np.ceil(np.sqrt(1 * n_epochs)).astype(int),
            np.ceil(np.sqrt(1 * n_epochs)).astype(int),
            i + 1
        )
        plt.imshow(im[0], aspect='auto')
        plt.axis('off')
        plt.show()
        plt.pause(0.1)

    # save our model
    torch.save(vae.state_dict(), 'vae.pth')
    plt.savefig('vae_training.png')
    plt.show()


if __name__ == '__main__':
    train_vae()