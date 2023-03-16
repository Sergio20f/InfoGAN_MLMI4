import torch
import numpy as np


def to_onehot(x, num_classes=10):
    """
    Converts an integer or a LongTensor to a one-hot tensor representation.

    Args:
        x (int or LongTensor): the class indices to be converted.
        num_classes (int): the total number of classes.

    Returns:
        A LongTensor of size (x.size(0), num_classes) if x is a LongTensor,
        or a LongTensor of size (1, num_classes) if x is an integer.
    """
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    
    if isinstance(x, int):
        # create a LongTensor of size (1, num_classes) and set the element at index x to 1
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
    else:
        # create a LongTensor of size (x.size(0), num_classes) and set all elements to 0
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes)
        c.zero_()
        # set the elements at the indices specified by x to 1 using the scatter_() function
        c.scatter_(1, x, 1) # dim, index, src value
    
    return c


def sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=None, supervised=False):
    """
    Generates random noise vectors and latent codes to be used as input to a generative model.

    Args:
        batch_size (int): the size of the batch.
        n_noise (int): the size of the noise vector.
        n_c_discrete (int): the number of categories for the discrete latent code.
        n_c_continuous (int): the size of the continuous latent code.
        label (int or None): the labels for the categorical latent code.
        supervised (bool): whether to use supervised or unsupervised learning.

    Returns:
        A tuple (z, c) containing:
        - z (FloatTensor): a random noise vector of size (batch_size, n_noise).
        - c (FloatTensor): a concatenated latent code of size (batch_size, n_c_discrete+n_c_continuous).
          The first n_c_discrete columns correspond to the one-hot encoded categorical latent code,
          while the last n_c_continuous columns correspond to the continuous latent code.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z = torch.randn(batch_size, n_noise).to(DEVICE)
    if supervised:
        # generate a one-hot encoded categorical latent code from the label
        c_discrete = to_onehot(label).to(DEVICE) # (B,10)
    else:
        # generate a one-hot encoded categorical latent code with random integers
        c_discrete = to_onehot(torch.LongTensor(batch_size, 1).random_(0, n_c_discrete)).to(DEVICE) # (B,10)
    # generate a continuous latent code with values between -1 and 1
    c_continuous = torch.zeros(batch_size, n_c_continuous).uniform_(-1, 1).to(DEVICE) # (B,2)
    # concatenate the categorical and continuous latent codes along the second dimension
    c = torch.cat((c_discrete.float(), c_continuous), 1)
    
    return z, c


def vanilla_gan_get_sample_image(G, n_noise):
    """
        Generates and saves 100 sample images from a generative model.
        Args:
            n_noise (int): integer representing the size of the noise vector.ñ
            G: the generator model.

    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z = torch.randn(100, n_noise).to(DEVICE)
    y_hat = G(z).view(100, 28, 28) # (100, 28, 28)
    result = y_hat.cpu().data.numpy()
    img = np.zeros([280, 280])
    for j in range(10):
        img[j*28:(j+1)*28] = np.concatenate([x for x in result[j*10:(j+1)*10]], axis=-1)
    return img


def get_sample_image(n_noise, n_c_continuous, G):
    """
    Generates and saves 100 sample images from a generative model.

    Args:
        n_noise (int): integer representing the size of the noise vector.ñ
        n_c_continuous (int): integer representing the size of the continuous latent code.
        G: the generator model.
    Returns:
        A tuple of two numpy arrays containing the sample images for the following categories:
        - Continuous code: 20 images of size (280, 280).
        - Discrete code: 10 images of size (280, 280).
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images = []
    
    # Generate images with varying continuous code
    for cc_type in range(2):
        for num in range(10):
            fix_z = torch.randn(1, n_noise)
            z = fix_z.to(DEVICE)
            cc = -1
            for i in range(10):
                cc += 0.2
                # One-hot encode the discrete code
                c_discrete = to_onehot(num).to(DEVICE) # (B,10)
                c_continuous = torch.zeros(1, n_c_continuous).to(DEVICE)
                # Set the value of the continuous code
                c_continuous.data[:, cc_type].add_(cc)
                # Combine the discrete and continuous codes
                c = torch.cat((c_discrete.float(), c_continuous), 1)
                # Generate an image from the combined code and the noise vector
                y_hat = G(z, c)
                # Concatenate the images horizontally
                line_img = torch.cat((line_img, y_hat.view(28, 28)), dim=1) if i > 0 else y_hat.view(28, 28)
            
            # Concatenate the rows of images vertically
            all_img = torch.cat((all_img, line_img), dim=0) if num > 0 else line_img

        # Convert the tensor to a numpy array
        img = all_img.cpu().data.numpy()
        images.append(img)
    
    # Generate images with varying discrete code
    for num in range(10):
        for i in range(10):
            # One-hot encode the discrete code
            c_discrete = to_onehot(i).to(DEVICE) # (B,10)
            z = torch.randn(1, n_noise).to(DEVICE)
            # Set the continuous code to zero
            c_continuous = torch.zeros(1, n_c_continuous).to(DEVICE)
            # Combine the discrete and continuous codes
            c = torch.cat((c_discrete.float(), c_continuous), 1)
            # Generate an image from the combined code and the noise vector
            y_hat = G(z, c)
            # Concatenate the images horizontally
            line_img = torch.cat((line_img, y_hat.view(28, 28)), dim=1) if i > 0 else y_hat.view(28, 28)
        
        # Concatenate the rows of images vertically
        all_img = torch.cat((all_img, line_img), dim=0) if num > 0 else line_img
    
    # Convert the tensor to a numpy array
    img = all_img.cpu().data.numpy()
    images.append(img)
    
    return images[0], images[1], images[2]


def log_gaussian(c, mu, var):
    """
        Criterion for Q(condition classifier)
    """
    return -((c - mu)**2)/(2*var+1e-8) - 0.5*torch.log(2*np.pi*var+1e-8)


def sample_c_discrete(n_samples, n_classes):
    """
    Sample c from the discrete latent space.
    """
    c = np.zeros((n_samples, n_classes))
    c[np.arange(n_samples), np.random.randint(0, n_classes, n_samples)] = 1
    return torch.tensor(c, dtype=torch.float32)


def fro_norm(model, eps=1e-8):
  params = model.parameters()
  total_norm = 0
  for p in params:
    if p.grad is None:
      continue
    param_norm = p.grad.data.norm(p='fro')
    total_norm += param_norm**2
  total_norm = total_norm**0.5
  return total_norm

def adaptive_clip_grad(M_model, G_norm, M_norm, eps=1e-8):
    parameters = [p for p in M_model.parameters() if p.requires_grad]
    if M_norm > 0.15:
      clip_grad(M_model)
      
    clip_coef = min(G_norm, M_norm)/(M_norm + eps)
    for p in parameters:
          p.grad.detach().mul_(clip_coef.to(p.grad.device))


def clip_grad(model, eps=1e-8):
    parameters = [p for p in model.parameters() if p.requires_grad]
    my_fro_norm = fro_norm(model)
    if my_fro_norm > 0.15:
      clip_coef = 0.15 / my_fro_norm
      for p in parameters:
        p.grad.detach().mul_(clip_coef.to(p.grad.device))