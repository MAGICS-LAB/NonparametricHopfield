from datasets.mnist_bags import MNISTBags
import torch.utils.data as DataLoader
from layers import *
from models import *
import wandb
from datasets import cifar_mem_retrieve, mnist_mem_retrieve, load_mnist
import torch.nn.functional as F
from math import sqrt
from torchvision.transforms.functional import to_pil_image
from scipy import spatial
import os

def man(X,z):

    X = torch.clamp(X, 0, 1)
    z = torch.clamp(z, 0, 1)
    return torch.sum(torch.abs(z - X),axis=1)/X.size(-1)

    # X, z: (batch_size, d_model)

def sqdiff(x, y):
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)
    sqdiff = torch.sum(torch.square(x - y), dim=-1)
    return torch.abs(sqdiff)/x.size(-1)

def cossim(x, y):
    return 1 - F.cosine_similarity(x, y)

class Exp:
    def __init__(self, config, i=0) -> None:
        self.config = config
        self.i = i
        data = self.config["data"]
        mem_size = self.config["mem_size"]
        self.path = f"results/memory_ret_{data}_{mem_size}_x"
        try:
            os.makedirs(self.path)
        except:
            pass
        run = wandb.init(
            # Set the project where this run will be logged
            project="Retrieval Experiments (1) "+ self.config["data"],
            # Track hyperparameters and run metadata
            config=self.config)

    def _get_data(self):

        if self.config["data"] == "cifar":
            return cifar_mem_retrieve(self.config["mem_size"])
        elif self.config["data"] == "mnist":
            return mnist_mem_retrieve(self.config["mem_size"])

    def _get_model(self):

        if self.config["mode"] != "favor":
            model = HopfieldLayer(  d_model=self.config["d_model"],
                                    n_heads=self.config["n_heads"], 
                                    update_steps=self.config["update_steps"], 
                                    mode=self.config["mode"],
                                    scale=self.config["scale"],
                                    prob=self.config["prob"])

        else:
            model = HopfieldLayer(  d_model=self.config["d_model"],
                                    n_heads=self.config["n_heads"], 
                                    update_steps=self.config["update_steps"], 
                                    mode=self.config["mode"],
                                    scale=self.config["scale"],
                                    prob=self.config["prob"],
                                    favor_mode=self.config["favor_mode"],
                                    mix=True,
                                    kernel_fn=self.config["kernel_fn"]
                                    )

        return model.cuda()

    def _get_cri(self):
        '''
        calculate memory retrieval error
        '''
        return man

    def half_img(self, imgs):

        # imgs: (M, c, h, w)
        imgs[:, :, :int(self.h/2), :] = 0.0
        return imgs

    def add_noise(self, img):
        sigma = self.config["noise"]
        perturb = torch.normal(0,sigma,size=[img.size(-1),])
        return torch.clamp(torch.abs(img + perturb),0,1)

    def retrieve(self):

        mem_patterns = self._get_data()
        mem_patterns = torch.stack(mem_patterns, dim=0)
        mem_size, self.c, self.h, self.w = mem_patterns.size()
        man_cor = 0
        sq_cor = 0

        if self.config["method"] == "half":
            queries = self.half_img(mem_patterns.clone())
        else:
            queries = self.add_noise(mem_patterns.clone())

        mem_patterns = mem_patterns.view(mem_size, -1).unsqueeze(0).cuda()
        queries = queries.view(mem_size, -1).unsqueeze(0).cuda()

        self.model = self._get_model()
        self.cri = self._get_cri()

        with torch.no_grad():

            out = self.model(queries, mem_patterns.clone()).squeeze(0)
            mem_patterns = mem_patterns.squeeze(0)
            # out: (M, d_model)
    
            man_dist = man(out, mem_patterns)
            sq_dist = sqdiff(out, mem_patterns)
            cos_dist = cossim(out, mem_patterns) 
            
            man_cor = (man_dist <= self.config["thr"]).sum().item()
            sq_cor = (sq_dist <= self.config["thr"]).sum().item()
            cos_cor = (cos_dist <= self.config["thr"]).sum().item()

            # print(man_cor, sq_cor, cos_cor)
            with open(f'{self.path}/man_dist{self.i}.npy', 'wb') as file:
                np.save(file, man_dist.cpu().numpy())

            with open(f'{self.path}/cos_dist{self.i}.npy', 'wb') as file:
                np.save(file, cos_dist.cpu().numpy())

            with open(f'{self.path}/sq_dist{self.i}.npy', 'wb') as file:
                np.save(file, sq_dist.cpu().numpy())

        wandb.log({"Man distance": man_dist.mean().item(), "square distance": sq_dist.mean().item(), "dot product distance": cos_dist.mean().item()})

    def visualize(self):
        
        '''
        visualize the retrieved image, with only 1 sample will be fine
        '''
        
        pass