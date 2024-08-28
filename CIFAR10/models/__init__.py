import torch
import numpy as np
from .min_norm_solver import MinNormSolver
from matplotlib import pyplot as plt

solver = MinNormSolver()
PATCH_SIZE = 8

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, test_pf=False):
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if test_pf:
            f1s = []
            f2s = []
            f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
            f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
            fs = np.stack([f1s, f2s])
            # print(f1s.shape, f2s.shape)
            # print(f1s, f2s)
            plt.scatter(f1s, f2s)
            plt.savefig('pf_ps{}_no_guidance.png'.format(PATCH_SIZE))   
            # f = img_patch_norm(x_mod, reduce=False).data.cpu().numpy()
            # plt.clf()
            # plt.scatter(np.arange(f.shape[0]), f)
            # plt.savefig('norm.png')      

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images
        
        
@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images
    
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

def get_constraint_grad(x, obj_idx=1):
    if obj_idx == 1:
        constraint_value = constraint_1(x)
    elif obj_idx == 2:
        constraint_value = constraint_2(x)
    else:
        pass
    constraint_grad = torch.autograd.grad(constraint_value.sum(), x, allow_unused=True, create_graph=True)[0]
    try:
        hessian = torch.autograd.grad(constraint_grad.sum(), x, allow_unused=True, retain_graph=False)[0]
    except:
        hessian = torch.zeros_like(x, device=x.device)
    return constraint_value, constraint_grad, hessian

def get_lambda_q(constraint_value, constraint_grad, hessian, log_prob_grad, step_size, eq=False):
    term1 = constraint_value/step_size
    term2 = torch.mean((log_prob_grad * constraint_grad)) + torch.mean(hessian)  
    term3 = torch.mean(torch.norm(constraint_grad)**2)
    lambda_q = (term1+term2)/(term3+1e-6)
    if not eq:
        lambda_q = torch.clamp(lambda_q, min=0.0) 
    return lambda_q
    
def anneal_Langevin_dynamics_single_constraint(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, obj_idx=1):
    images = []

    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_value, constraint_grad, hessian = get_constraint_grad(x_mod, obj_idx)
            lambda_q = get_lambda_q(constraint_value, constraint_grad, hessian, grad, step_size)
            grad = grad-lambda_q*constraint_grad

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)
            print(lambda_q, constraint_1(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
            
            del constraint_grad  # Delete the constraint_grad tensor
            del hessian
            torch.cuda.empty_cache()  # Empty the GPU cache

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images
 
def get_constraints_grad(x, return_constraint=False):
    constraint_value_1 = constraint_1(x, reduce=False)
    constraint_value_2 = constraint_2(x, reduce=False)
    constraint_grad_1 = torch.autograd.grad(constraint_value_1.mean(), x, allow_unused=True, create_graph=True)[0]
    constraint_grad_2 = torch.autograd.grad(constraint_value_2.mean(), x, allow_unused=True, create_graph=True)[0]
    if return_constraint:
        return constraint_grad_1, constraint_grad_2, constraint_value_1, constraint_value_2
    else:
        return constraint_grad_1, constraint_grad_2   

def pf_diversity(x, s=2, MAX=10000., EPS=0.0001):
    # M is the loss matrix (list): [[task1, task2, ...], [task1, task2, ...], ....]
    _, _, c1, c2 = get_constraints_grad(x, return_constraint=True)  
    M = torch.stack([c1, c2], dim=1)    
    diversity_loss = torch.cdist(M, M)**2
    diversity_loss = diversity_loss + MAX * torch.eye(diversity_loss.shape[0]).to(x.device)

    if s >= 10:
        diversity_loss = (1. / (EPS + diversity_loss)).max()
    else:
        diversity_loss = ((EPS + diversity_loss) ** (-s / 2.)).mean()
    diversity_grad = torch.autograd.grad(diversity_loss, x, allow_unused=True, create_graph=True)[0]

    return diversity_loss, diversity_grad

def single_obj_diversity(x, s=2, MAX=10000., EPS=0.0001):
    # M is the loss matrix (list): [[task1, task2, ...], [task1, task2, ...], ....]
    _, _, c1, c2 = get_constraints_grad(x, return_constraint=True)  
    M = c1+c2
    M = M.view(-1, 1)
    # diversity_loss = torch.zeros(len(M), len(M)).to(x.device)
    # for net_id1 in range(0,len(M)):
    #     for net_id2 in range(net_id1+1, len(M)):
    #         for task_id in range(len(M[0])):
    #             diversity_loss[net_id1, net_id2] += (M[net_id1][task_id] - M[net_id2][task_id])**2
    #             diversity_loss[net_id2, net_id1] += (M[net_id1][task_id] - M[net_id2][task_id])**2
    diversity_loss = torch.cdist(M, M)**2
    diversity_loss = diversity_loss + MAX * torch.eye(diversity_loss.shape[0]).to(x.device)

    if s >= 10:
        diversity_loss = (1. / (EPS + diversity_loss)).max()
    else:
        diversity_loss = ((EPS + diversity_loss) ** (-s / 2.)).mean()
    diversity_grad = torch.autograd.grad(diversity_loss, x, allow_unused=True, create_graph=True)[0]

    return diversity_loss, diversity_grad

def mgd(task_grad, alpha=None, return_alpha=False):
    """
    implemented with Frank wolfe
    task_grad: num_task, num_particle, num_var
    alpha: num_task, num_particle
    """
    grad_shape = task_grad.shape
    task_grad = task_grad.view(grad_shape[0], grad_shape[1], -1)
    max_iter = 30
    if alpha is None:
        # initialize
        alpha = torch.zeros(task_grad.shape[0], task_grad.shape[1]).to(task_grad.device)
        norm = (task_grad**2).sum(axis=2) # num_task, num_particle
        id = norm.argmin(axis=0)
        alpha[id, torch.arange(0, task_grad.shape[1]).to(task_grad.device)] += 1.

    grad = (torch.unsqueeze(alpha, 2)*task_grad).sum(axis=0) # num_particle, num_var
    for _ in range(max_iter):
        product = (task_grad * grad).sum(axis=2) # num_task, num_particle
        id = product.argmin(axis=0) # num_task
        # selected_grad = task_grad[id, torch.arange(0, task_grad.shape[1]), :] # num_particle, num_var
        # delta = selected_grad - grad
        # todo: check lr!
        # lr = -(grad * delta).sum(axis=1) / (delta**2).sum(axis=1) # num_particle
        lr = 1./(1.+ _)
        incre = torch.zeros_like(alpha)
        incre[id, torch.arange(0, task_grad.shape[1])] += 1.
        alpha = alpha * (1. - lr) + lr * incre
        grad = (torch.unsqueeze(alpha, 2)*task_grad).sum(axis=0)
    grad = grad.view(grad_shape[1], grad_shape[2], grad_shape[3], grad_shape[4])
    if return_alpha:
        return grad, alpha
    else:
        return grad
        
def energy_fn(x, pref):
    _, _, c1, c2 = get_constraints_grad(x, return_constraint=True)  
    F = torch.stack([c1, c2], dim=1)    
    n_task = len(pref)
    pref = pref.view(1, -1).to(F.device)
    energy = (F*pref).mean(dim=0)
    # energy = torch.zeros(n_task)
    # for task_id in range(n_task):
    #     energy[task_id] += F[:, task_id] * pref[task_id]
    energy /= energy.sum()
    energy = (torch.log(n_task * energy) * energy).sum()

    # x.grad.data.zero_()
    # energy.backward()
    # energy_grad = x.grad.clone().detach()
    energy_grad = torch.autograd.grad(energy, x, allow_unused=True, create_graph=True)[0]

    return energy, energy_grad

def project_to_linear_span(g, G, epsilon=1e-6, normalize=False, constrain=None):
    """
    :param g: num_particle, num_var
    :param G: num_task, num_particle, num_var
    :return: for each particle, get the projection on linear spam of G
    """
    if normalize:
        G_norm = ((G**2).sum(axis=1, keepdim=True).sqrt())
        g_norm = ((g**2).sum(axis=0, keepdim=True).sqrt())
        G = G/(G_norm+epsilon)
        g = g/(g_norm+epsilon)
    G_shape = G.shape
    g = g.view(g.shape[0], -1)
    G = G.view(G.shape[0], G.shape[1], -1)
    g_project = torch.zeros_like(g)
    n_task = G.shape[0]
    n_particle = G.shape[1]
    betas = torch.zeros(n_task, n_particle).to(G.device)
    for _ in range(g.shape[0]):
        #beta, LU = torch.solve(G[:, _, :].mm(g[[_], :].T), G[:, _, :].mm(G[:, _, :].T) + epsilon * torch.eye(G.shape[0]))
        if constrain is None:
            beta = torch.linalg.solve(G[:, _, :].mm(G[:, _, :].T) + epsilon * torch.eye(G.shape[0]).to(G.device), G[:, _, :].mm(g[[_], :].T))
        else:
            beta = torch.linalg.solve(G[:, _, :].mm(G[:, _, :].T) + epsilon * torch.eye(G.shape[0]).to(G.device), G[:, _, :].mm(g[[_], :].T)+constrain[[_]].unsqueeze(0).repeat(2, 1))
        g_project[_, :] = (beta.T).mm(G[:, _, :]).squeeze()

        betas[:, [_]] = beta

    g_project = g_project.view(G_shape[1], G_shape[2], G_shape[3], G_shape[4])
    return g_project, betas
       
def grad_search_proud_diversity(logp_grads, diversity_grad, task_grads, alpha=0.5, threshold=0.01, lambda_d=5, normalize=False):
    """
    energy_grad: num_particle, num_var; torch.tensor
    task_grad: num_task, num_particle, num_var; torch.tensor
    num_particle = 1
    """    
    if normalize:
        logp_norm = (logp_grads**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        logp_grads = logp_grads/logp_norm
        task_norm = (task_grads**2).sum(axis=[2, 3, 4], keepdim=True).sqrt()
        #task_norm = task_norm.mean(axis=0, keepdim=True)   # using this can keep the scale of each objective
        task_grads = task_grads/task_norm
        diversity_norm = (diversity_grad**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        diversity_grad = diversity_grad/diversity_norm
    
    diversity_grad = diversity_grad*lambda_d
        
    n_particle = task_grads.shape[1]

    # obtain mgd gradient
    #d_mgd, betas = self.solve_min_norm_2_loss(task_grads[0], task_grads[1], return_beta=True)
    d_mgd, betas = mgd(task_grads, return_alpha=True)
    d_mgd_norm = (d_mgd ** 2).sum(axis=[1, 2, 3])
    print(d_mgd_norm[:5])
    # constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold)
    constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold * (task_grads**2).sum(axis=[2, 3, 4]).mean())
    constrain = constrain
    betas_mask = (d_mgd_norm > threshold).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

    #grad_project, betas = project_to_linear_span(-logp_grads, task_grads, normalize=False, constrain=None)  # betas: n_task, n_particle
    betas = betas * (betas >= 0.)
    print(betas.squeeze()[0][:5])
    betas = betas.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    betas.requires_grad = True

    lr = 0.1 #step_size
    for _ in range(100):
        loss = 0.5 * (((task_grads * betas).sum(axis=0) + (logp_grads+diversity_grad)) ** 2).sum() / n_particle \
                - (betas.sum(axis=0).squeeze() * constrain).sum()
        # loss = 0.5 * (((task_grads * betas).sum(axis=0) + (logp_grads)) ** 2).sum() \
        #         - (betas.sum(axis=0).squeeze() * constrain).sum()
        if _ == 0:
            print(loss)
        loss.backward()
        betas.data -= lr * betas.grad.data
        betas.grad.data.zero_()
        betas.data *= (betas.data >= 0.)
    print(loss)
    print(betas.squeeze()[0, :5])
    grad_constrain = ((task_grads * (betas * betas_mask)).sum(axis=0) + (logp_grads+diversity_grad)).detach().clone()
    
    if normalize:
        grad_constrain = grad_constrain*logp_norm
    return grad_constrain, betas * betas_mask

def anneal_Langevin_dynamics_proud_diverisity(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
        final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False, lambda_diver=0.):
    images = []
    
    alpha = 0.5
    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_grad_1, constraint_grad_2 = get_constraints_grad(x_mod)
            constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
            f_grads = constraint_grads.data.detach().clone()
            kl_grad = grad.data.detach().clone()
            diversity_loss, diversity_grad = pf_diversity(x_mod)
            # energy_fn(x_mod, torch.tensor([0.1, 0.9]).float()) 
            diversity_grad = diversity_grad.data.detach().clone()
            grad, _ = grad_search_proud_diversity(-kl_grad, diversity_grad, f_grads, alpha=alpha, threshold=3e-2, \
                lambda_d=lambda_diver, normalize=True)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod - step_size * grad + noise * np.sqrt(step_size * 2)
            
            print(constraint_1(x_mod), constraint_2(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        with torch.no_grad():
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        plt.scatter(f1s, f2s)
        plt.savefig('pf_ps{}_proud_diversity{}.png'.format(PATCH_SIZE, lambda_diver, alpha))     
                    
    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images

def grad_search_mgd_diversity(logp_grads, diversity_grad, task_grads, alpha=0.5, threshold=0.01, \
    lambda_diver=5, lambda_mgd=1, normalize=False):
    """
    energy_grad: num_particle, num_var; torch.tensor
    task_grad: num_task, num_particle, num_var; torch.tensor
    num_particle = 1
    """
    d_mgd, betas = mgd(task_grads, return_alpha=True)
    print(betas[0][:8])
    if normalize:
        logp_norm = (logp_grads**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        logp_grads = logp_grads/logp_norm
        d_mgd_norm = (d_mgd ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        d_mgd = d_mgd/d_mgd_norm
        print(d_mgd_norm.squeeze()[:5])
        diversity_norm = (diversity_grad**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        diversity_grad = diversity_grad/diversity_norm
    
    diversity_grad = diversity_grad*lambda_diver
    
    grad_constrain = (d_mgd*lambda_mgd + logp_grads+diversity_grad).detach().clone()
    
    if normalize:
        grad_constrain = grad_constrain*logp_norm
    return grad_constrain
    
def anneal_Langevin_dynamics_mgd_diverisity(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
        final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False, lambda_diver=0, \
            lambda_mgd=1):
    
    images = []
    f_grads_normalize = False

    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_grad_1, constraint_grad_2, c1, c2 = get_constraints_grad(x_mod, return_constraint=True)
            if f_grads_normalize:
                c1 = c1.view(-1, 1, 1, 1)
                c2 = c2.view(-1, 1, 1, 1)
                constraint_grad_1 = constraint_grad_1/c1
                constraint_grad_2 = constraint_grad_2/c2
            constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
            f_grads = constraint_grads.data.detach().clone()
            kl_grad = grad.data.detach().clone()
            diversity_loss, diversity_grad = pf_diversity(x_mod)
            diversity_grad = diversity_grad.data.detach().clone()
            grad = grad_search_mgd_diversity(-kl_grad, diversity_grad, f_grads, alpha=0.5, threshold=1e-2, lambda_diver=lambda_diver, \
                lambda_mgd=lambda_mgd, normalize=True)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod - step_size * grad + noise * np.sqrt(step_size * 2)
            
            print(constraint_1(x_mod), constraint_2(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        with torch.no_grad():
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        plt.scatter(f1s, f2s)
        plt.savefig('pf_ps{}_mgd{}_diveristy{}.png'.format(PATCH_SIZE, lambda_mgd, lambda_diver))  
        print('pf_ps{}_mgd{}_diveristy{}.png'.format(PATCH_SIZE, lambda_mgd, lambda_diver))  
                    
    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images

def grad_search_mgd_only(logp_grads, diversity_grad, task_grads, alpha=0.5, threshold=0.01, \
    lambda_diver=1, normalize=False):
    """
    energy_grad: num_particle, num_var; torch.tensor
    task_grad: num_task, num_particle, num_var; torch.tensor
    num_particle = 1
    """
    if normalize:
        logp_norm = (logp_grads**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        logp_grads = logp_grads/logp_norm
        task_norm = (task_grads**2).sum(axis=[2, 3, 4], keepdim=True).sqrt()
        # task_norm = task_norm.mean(axis=0, keepdim=True)
        task_grads = task_grads/task_norm
        diversity_norm = (diversity_grad**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        diversity_grad = diversity_grad/diversity_norm
    
    d_mgd, betas = mgd(task_grads, return_alpha=True)
    print(betas[0][:8])
    grad_constrain = (d_mgd+diversity_grad*lambda_diver).detach().clone()
    
    if normalize:
        grad_constrain = grad_constrain*logp_norm
    return grad_constrain
    
def mgd_only_diversity(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
        final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False, lambda_diver=0):
    images = []

    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_grad_1, constraint_grad_2 = get_constraints_grad(x_mod)
            constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
            f_grads = constraint_grads.data.detach().clone()
            kl_grad = grad.data.detach().clone()
            diversity_loss, diversity_grad = pf_diversity(x_mod)
            diversity_grad = diversity_grad.data.detach().clone()
            grad = grad_search_mgd_only(-kl_grad, diversity_grad, f_grads, alpha=0.5, threshold=1e-2, \
                lambda_diver=lambda_diver, normalize=True)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod - step_size * grad + noise * np.sqrt(step_size * 2)*0
            
            print(constraint_1(x_mod), constraint_2(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        with torch.no_grad():
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        plt.scatter(f1s, f2s)
        plt.savefig('pf_ps{}_mgd_only_diversity{}.png'.format(PATCH_SIZE, lambda_diver))     
                    
    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images
        
def mgd_only_diversity_(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
        final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False, lambda_diver=0):
    images = []
    normalize = False

    n_steps = 1000
    step_size = 1

    x_mod.requires_grad = True
    for i in range(n_steps):
        constraint_grad_1, constraint_grad_2 = get_constraints_grad(x_mod)
        constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
        f_grads = constraint_grads.data.detach().clone()
        if normalize:
            fs_norm = (f_grads**2).sum(axis=[2, 3, 4], keepdim=True).sqrt()
            f_grads = f_grads/fs_norm
        d_mgd, betas = mgd(f_grads, return_alpha=True)
        print(betas[0][:5])
        diversity_loss, diversity_grad = pf_diversity(x_mod)
        diversity_grad = diversity_grad.data.detach().clone()
        grad = d_mgd+lambda_diver*diversity_grad
        x_mod = x_mod - step_size * d_mgd 
        print(constraint_1(x_mod), constraint_2(x_mod))
        
    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        plt.scatter(f1s, f2s)
        plt.savefig('pf_ps{}_mgd_only_diversity{}.png'.format(PATCH_SIZE, lambda_diver))   

    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images

def grad_search_m1_mgd(logp_grads, task_grads, alpha=0.5, threshold=0.01, \
        normalize=False):
    """
    energy_grad: num_particle, num_var; torch.tensor
    task_grad: num_task, num_particle, num_var; torch.tensor
    num_particle = 1
    """

    if normalize:
        logp_norm = (logp_grads**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        logp_grads = logp_grads/logp_norm
        task_norm = (task_grads**2).sum(axis=[2, 3, 4], keepdim=True).sqrt()
        # task_norm = task_norm.mean(axis=0, keepdim=True)
        task_grads = task_grads/task_norm
        
    # obtain mgd gradient
    alphass = []
    for i in range(logp_grads.shape[0]):
        all_grad = [task_grads[j] for j in range(task_grads[:, i, :, :, :].shape[0])]
        all_grad.append(logp_grads[i])
        alphas, _ = solver.find_min_norm_element(all_grad)
        alphass.append(alphas)
    alphas = np.stack(alphass, axis=1)
    alphas = torch.from_numpy(alphas).to(logp_grads.device).unsqueeze(2).unsqueeze(3).unsqueeze(4).float()
    all_grads = torch.cat([task_grads, logp_grads.unsqueeze(0)], dim=0)
    grad_constrain = torch.sum(all_grads*alphas, dim=0)
    
    if normalize:
        grad_constrain = grad_constrain*logp_norm
    return grad_constrain
    
def anneal_Langevin_dynamics_m1_mgd(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008, \
        final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False):
    images = []

    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_grad_1, constraint_grad_2, c1, c2 = get_constraints_grad(x_mod, return_constraint=True)
            constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
            f_grads = constraint_grads.data.detach().clone()
            kl_grad = grad.data.detach().clone()
            grad = grad_search_m1_mgd(-kl_grad, f_grads, alpha=0.5, threshold=1e-2, normalize=True)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod - step_size * grad + noise * np.sqrt(step_size * 2)
            
            print(constraint_1(x_mod), constraint_2(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        with torch.no_grad():
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        # print(f1s.shape, f2s.shape)
        # print(f1s, f2s)
        plt.scatter(f1s, f2s)
        plt.savefig('pf_ps{}_m1_mgd.png'.format(PATCH_SIZE))   
        # f = img_patch_norm(x_mod, reduce=False).data.cpu().numpy()
        # plt.clf()
        # plt.scatter(np.arange(f.shape[0]), f)
        # plt.savefig('norm.png')      
                    
    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images

def grad_search_linear_sum_diversity(logp_grads, diversity_grad, task_grads, \
    lambda_diver=5, lambda_f=1, normalize=False, w1=0.5):
    """
    energy_grad: num_particle, num_var; torch.tensor
    task_grad: num_task, num_particle, num_var; torch.tensor
    num_particle = 1
    """
    if normalize:
        logp_norm = (logp_grads**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        logp_grads = logp_grads/logp_norm
        task_norm = (task_grads**2).sum(axis=[2, 3, 4], keepdim=True).sqrt()
        # task_norm = task_norm.mean(axis=0, keepdim=True)
        task_grads = task_grads/task_norm
        diversity_norm = (diversity_grad**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        diversity_grad = diversity_grad/diversity_norm
    
    if lambda_diver == 0:
        diversity_grad = 0
    else:
        diversity_grad = diversity_grad*lambda_diver
    grad_constrain = ((w1*task_grads[0]+(1-w1)*task_grads[1])*lambda_f+logp_grads+diversity_grad).detach().clone()
    if normalize:
        grad_constrain = grad_constrain*logp_norm
    return grad_constrain

def anneal_Langevin_dynamics_linear_sum_diversity(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008, \
        final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False, lambda_diver = 0, lambda_f = 1, w1=0.5):
    images = []    

    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_grad_1, constraint_grad_2 = get_constraints_grad(x_mod)
            constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
            f_grads = constraint_grads.data.detach().clone()
            kl_grad = grad.data.detach().clone()
            diversity_loss, diversity_grad = single_obj_diversity(x_mod)
            diversity_grad = diversity_grad.data.detach().clone()
            grad = grad_search_linear_sum_diversity(-kl_grad, diversity_grad, f_grads, lambda_diver=lambda_diver, \
                lambda_f=lambda_f, normalize=True, w1=w1)
            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod - step_size * grad + noise * np.sqrt(step_size * 2)
            
            print(constraint_1(x_mod), constraint_2(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        with torch.no_grad():
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        plt.scatter(f1s, f2s)
        plt.savefig('pf_ps{}_linear_sum{}_diversity{}.png'.format(PATCH_SIZE, lambda_f, lambda_diver))   
                    
    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images

def grad_search_b_kl_mgd_diversity(logp_grads, diversity_grad, task_grads, alpha=0.5, threshold=0.01, \
    lambda_diver=1, normalize=False):
    """
    energy_grad: num_particle, num_var; torch.tensor
    task_grad: num_task, num_particle, num_var; torch.tensor
    num_particle = 1
    """

    if normalize:
        logp_norm = (logp_grads**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        logp_grads = logp_grads/logp_norm
        task_norm = (task_grads**2).sum(axis=[2, 3, 4], keepdim=True).sqrt()
        #task_norm = task_norm.mean(axis=0, keepdim=True)
        task_grads = task_grads/task_norm
        diversity_norm = (diversity_grad**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        diversity_grad = diversity_grad/diversity_norm
    
    n_particle = task_grads.shape[1]

    # obtain mgd gradient
    #d_mgd, betas = self.solve_min_norm_2_loss(task_grads[0], task_grads[1], return_beta=True)
    d_mgd = mgd(task_grads)
    d_mgd_norm = (d_mgd ** 2).sum(axis=[1, 2, 3])
    print(d_mgd_norm[:5])
    #constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold)
    constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold * (task_grads**2).sum(axis=[2, 3, 4]).mean())
    constrain = constrain
    betas_mask = (d_mgd_norm > threshold).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    
    lambda_q = ((-(logp_grads+diversity_grad*lambda_diver)*d_mgd).sum(axis=[1, 2, 3])+constrain)/(d_mgd_norm+1e-6)
    lambda_q = torch.clamp(lambda_q, min=0.0)
    print(lambda_q[:5])   
    lambda_q = lambda_q.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    
    grad_constrain = logp_grads+diversity_grad+d_mgd*(lambda_q*betas_mask)
    
    if normalize:
        grad_constrain = grad_constrain*logp_norm
    return grad_constrain, lambda_q * betas_mask

def anneal_Langevin_dynamics_b_kl_mgd_diverisity(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False):
    images = []
    lambda_diver = 0
    
    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_grad_1, constraint_grad_2 = get_constraints_grad(x_mod)
            constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
            f_grads = constraint_grads.data.detach().clone()
            kl_grad = grad.data.detach().clone()
            diversity_loss, diversity_grad = pf_diversity(x_mod)
            diversity_grad = diversity_grad.data.detach().clone()
            grad, _ = grad_search_b_kl_mgd_diversity(-kl_grad, diversity_grad, f_grads, alpha=0.5, threshold=1e-2, lambda_diver=lambda_diver, normalize=True)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod - step_size * grad + noise * np.sqrt(step_size * 2)
            
            print(constraint_1(x_mod), constraint_2(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        with torch.no_grad():
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        plt.scatter(f1s, f2s)
        plt.savefig('pf_b_kl_mgd_diversity{}.png'.format(lambda_diver))   
                    
    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images
    
def grad_search_b_mgd_kl_diversity(logp_grads, diversity_grad, task_grads, alpha=0.5, threshold=0.01, \
    lambda_diver=1, normalize=False):
    """
    energy_grad: num_particle, num_var; torch.tensor
    task_grad: num_task, num_particle, num_var; torch.tensor
    num_particle = 1
    """

    if normalize:
        logp_norm = (logp_grads**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        logp_grads = logp_grads/logp_norm
        task_norm = (task_grads**2).sum(axis=[2, 3, 4], keepdim=True).sqrt()
        task_grads = task_grads/task_norm
        diversity_norm = (diversity_grad**2).sum(axis=[1, 2, 3], keepdim=True).sqrt()
        diversity_grad = diversity_grad/diversity_norm
    
    n_particle = task_grads.shape[1]

    # obtain mgd gradient
    #d_mgd, betas = self.solve_min_norm_2_loss(task_grads[0], task_grads[1], return_beta=True)
    d_mgd = mgd(task_grads)
    d_mgd_norm = (d_mgd ** 2).sum(axis=[1, 2, 3])
    print(d_mgd_norm[:5])
    constrain = alpha * logp_norm * (logp_norm > threshold)
    constrain = constrain.squeeze()
    betas_mask = (logp_norm > threshold)
    
    lambda_q = ((-(logp_grads+diversity_grad*lambda_diver)*d_mgd).sum(axis=[1, 2, 3])+constrain)
    lambda_q = torch.clamp(lambda_q, min=0.0)
    # print((-energy_grads*d_mgd).sum(axis=[1, 2, 3])[:5])
    print(lambda_q[:5])   
    lambda_q = lambda_q.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    
    grad_constrain = (logp_grads+diversity_grad)*(lambda_q*betas_mask)+d_mgd
    
    if normalize:
        grad_constrain = grad_constrain*logp_norm
    return grad_constrain, lambda_q * betas_mask

def anneal_Langevin_dynamics_b_mgd_kl_diverisity(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, obj_idx=1, test_pf=False):
    images = []
    lambda_diver = 0
    
    x_mod.requires_grad = True
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(x_mod, labels)
            
            constraint_grad_1, constraint_grad_2 = get_constraints_grad(x_mod)
            constraint_grads = torch.stack([constraint_grad_1, constraint_grad_2])
            f_grads = constraint_grads.data.detach().clone()
            kl_grad = grad.data.detach().clone()
            diversity_loss, diversity_grad = pf_diversity(x_mod)
            diversity_grad = diversity_grad.data.detach().clone()
            grad, _ = grad_search_b_mgd_kl_diversity(-kl_grad, diversity_grad, f_grads, alpha=0.5, threshold=1e-2, lambda_diver=lambda_diver, normalize=True)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod - step_size * grad + noise * np.sqrt(step_size * 2)
            
            print(constraint_1(x_mod), constraint_2(x_mod))

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        with torch.no_grad():
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if test_pf:
        f1s = []
        f2s = []
        f1s = constraint_1(x_mod, reduce=False).data.cpu().numpy()
        f2s = constraint_2(x_mod, reduce=False).data.cpu().numpy()
        fs = np.stack([f1s, f2s])
        plt.scatter(f1s, f2s)
        plt.savefig('pf_b_mgd_kl_diversity{}.png'.format(lambda_diver))   
                    
    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images     
    
                