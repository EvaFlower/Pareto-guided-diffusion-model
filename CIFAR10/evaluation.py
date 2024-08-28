import numpy as np
from torchvision.datasets import CIFAR10
import os
import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import fid 
from PIL import Image
from matplotlib import pyplot as plt
from pymoo.indicators.hv import HV
import scipy
from torchvision.utils import make_grid, save_image
from botorch.utils.multi_objective import is_non_dominated


def hypervolumn(A, ref=None, type='acc'):
    """
    :param A: np.array, num_points, num_task
    :param ref: num_task
    """
    dim = A.shape[1]        

    if type == 'acc':
        if ref is None:
            ref = np.zeros(dim)
        hv = HV(ref_point=ref)
        return hv(-A)

    elif type == 'loss':
        if ref is None:
            ref = np.ones(dim)
        hv = HV(ref_point=ref)
        return hv(A)
    else:
        print('type not implemented')
        return None

def load_images_from_directory(directory_path):
    images = []
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return images

    # Iterate over files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a supported image format
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Construct the full path to the image file
            image_path = os.path.join(directory_path, filename)
            
            # Open the image using Pillow
            try:
                img = Image.open(image_path)
                img = np.array(img)
                images.append(img)
                #print(f"Loaded image: {filename}")
            except Exception as e:
                print(f"Error loading image {filename}: {str(e)}")
    images = np.stack(images, axis=0)
    return images

PATCH_SIZE = 8
def constraint_1(x, patch_size=PATCH_SIZE, bound=1e-4, reduce=True, con_t=1):
    begin_index =int((32-patch_size)/2-1)
    patch_x = x[:, :, begin_index:begin_index+patch_size, begin_index:begin_index+patch_size]
    if reduce:
        var = torch.mean((patch_x-con_t).pow(2))-bound
    else:
        var = torch.mean((patch_x-con_t).pow(2), dim=(1, 2, 3))-bound
    return var

def constraint_2(x, patch_size=PATCH_SIZE, bound=1e-4, reduce=True, con_t=0.5):
    begin_index =int((32-patch_size)/2-1)
    patch_x = x[:, :, begin_index:begin_index+patch_size, begin_index:begin_index+patch_size]
    if reduce:
        var = torch.mean((patch_x-con_t).pow(2))-bound
    else:
        var = torch.mean((patch_x-con_t).pow(2), dim=(1, 2, 3))-bound
    return var

method = 'ps8_linear_sum1_w0.8'
img_path = './exp/image_samples/images_{}/samples_300000.pth'.format(method)
samples = torch.load(img_path)

# img_path = '/home/yinghua/project/ncsnv2/exp/image_samples/images_{}'.format(method)
# samples = load_images_from_directory(img_path)
# samples = samples.transpose(0, 3, 1, 2)
# samples = torch.from_numpy(samples).float()/255.

print(samples.shape, torch.min(samples), torch.max(samples))
image_grid = make_grid(samples[:25], 5)
save_image(image_grid, 'image_grid_{}.png'.format(method))

f1s = constraint_1(samples, reduce=False).data.cpu().numpy()
f2s = constraint_2(samples, reduce=False).data.cpu().numpy()
fs = np.stack([f1s, f2s], axis=1)
ref_point = np.array([0.25, 0.25])

pf_mask = is_non_dominated(torch.tensor(-fs))
pf = fs[pf_mask]

hv_targets = hypervolumn(fs, ref_point, 'loss')
print('hv for targets: ', hv_targets)

plt.scatter(f1s, f2s)
plt.savefig('pf_{}_2.png'.format(method))
np.save('pf_{}.npy'.format(method), fs)

scipy.io.savemat('pf_{}.mat'.format(method), {'fs': fs})

        
# final_gen_samples = samples.data.mul_(255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).numpy()
# final_gen_samples = final_gen_samples.transpose(0, 2, 3, 1)
method = 'ps8_linear_sum1_diversity0_50k'
img_path = './exp/image_samples/images_{}'.format(method)
final_gen_samples = load_images_from_directory(img_path)
print('gen:', final_gen_samples.shape, np.max(final_gen_samples), np.min(final_gen_samples))

root_path = './exp'
test_dataset = CIFAR10(os.path.join(root_path, 'datasets', 'cifar10'), train=True, download=False,
                                   transform=None)
train_data = test_dataset.data
print('training data', train_data.shape, np.max(train_data), np.min(train_data))

inception_path = fid.check_or_download_inception(None) # download inception network
fid.create_inception_graph(inception_path)
                
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_real, sigma_real = fid.calculate_activation_statistics(train_data, sess, batch_size=100) 
print('estimate real mu sigma done!')
# np.save('fid_mu_real.npy', mu_real)
# np.save('fid_sigma_real.npy', sigma_real)
                    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(final_gen_samples, sess, batch_size=100)
print('estimate gen mu sigma done!')

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)

