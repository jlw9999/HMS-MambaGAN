import sys
sys.path.insert(0, r'/HOME/scw6d2x/run/jlw')
import random
import itertools
import functools
from HMS_MambaGAN.models.base_model import BaseModel, get_norm_layer, init_net, get_scheduler
from HMS_MambaGAN.models.GAN_Loss import *
from HMS_MambaGAN.models.Diffusion import DiffusionModel
from .GLGCM_Loss import GLGCM_Loss, OutlineROI
from .networks import ResnetGenerator
from HMS_MambaGAN.models.modules import HMSMamba
from . import nn_configs as configs

class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        gpu_ids=[],
    ):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            print(self.model(input).size())
            return self.model(input)

def define_G(input_nc,output_nc, ngf, which_model_netG,vit_name,img_size,pre_trained_path, norm="batch", use_dropout=False,init_type="normal",  gpu_ids=[],pre_trained_trans=True,pre_trained_resnet=0,):
    norm_layer = get_norm_layer(norm_type=norm)
    print("******************",which_model_netG)
    if which_model_netG == "ResnetGenerator":
        net = ResnetGenerator(input_nc, output_nc, 64, n_blocks = 9, n_attentions=5, argmax=False)
        netG = init_net(net, init_type, gpu_ids)
    
    elif which_model_netG == "HMSMambaGAN":
        print(vit_name)
        netG = HMSMamba(
            modules.CONFIGS[vit_name],
            input_dim=input_nc,
            img_size=img_size,
            output_dim=1,
            vis=False,
        )
        config_vit = modules.CONFIGS[vit_name]
        if pre_trained_resnet:
            pre_trained_model = modules.ResCNN(
                modules.CONFIGS[vit_name],
                input_dim=input_nc,
                img_size=img_size,
                output_dim=1,
                vis=False,
            )
            save_path = pre_trained_path
            print("pre_trained_path: ", save_path)
            pre_trained_model.load_state_dict(torch.load(save_path))

            pretrained_dict = pre_trained_model.state_dict()
            model_dict = netG.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            netG.load_state_dict(model_dict)

        if pre_trained_trans:
            for param in netG.parameters():
                if param.requires_grad:
                    init.normal_(
                        param.data, mean=0, std=0.02
                    )  # Randomly initialize with normal distribution

    return netG

def define_Diff_G(conf):
    netG = DiffusionModel(conf).cuda()
    return netG

def define_D(in_channel, ndf, n_layers_D = 3, norm='batch', init_type='normal', use_sigmoid=False,init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    # net = NLayerDiscriminator(in_channel, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    net = NLayerDiscriminator(in_channel,ndf,n_layers=n_layers_D,norm_layer=norm_layer,use_sigmoid=use_sigmoid,gpu_ids=gpu_ids,)
    return init_net(net, init_type, init_gain, gpu_ids)

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

# **********************************************  HMS_MambaGAN_HMS_Module  *********************************************
class HMS_GANModel_hms(BaseModel):
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_GAN', 'G_L1', 'G_HisGDL', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.schedulers = [get_scheduler(optimizer, conf) for optimizer in self.optimizers]

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = define_G(
            conf.input_nc,
            conf.output_nc,
            conf.ngf,
            conf.which_model_netG,
            conf.vit_name,
            conf.fineSize,
            conf.pre_trained_path,
            conf.norm,
            not conf.no_dropout,
            conf.init_type,
            self.gpu_ids,
            pre_trained_trans=conf.pre_trained_transformer,
            pre_trained_resnet=conf.pre_trained_resnet,
        ).cuda()
        self.emaG = EMA(self.netG, 0.9999)
        self.emaG.register()

        if self.isTrain:
            self.netD = define_D(conf.in_channel + conf.out_channel, ndf=64, n_layers_D=3, norm='instance',init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionGDL = GDLoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionOutlineROI = OutlineROI().to(self.device)
            self.criterionGLGCM = GLGCM_Loss().to(self.device)
            self.criterionVGG = VGGLoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.fake_B_pool = ImagePool(50)
            self.alpha = [0.0]

    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.original_A = input['original_A' if task else 'original_B'].to(self.device)

        self.real_B = input['B' if task else 'A'].to(self.device)
        self.original_B = input['original_B' if task else 'original_A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']
        self.image_name = input['name']

    def forward(self):
        self.fake_B = self.netG(self.real_A.cuda()).cuda()

    def backward_D(self):
        fake_B = self.fake_B_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B, fake_B) * 10.0

        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_GP

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = 10.0 * self.criterionL1(self.fake_B, self.real_B).to(self.device)

        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0
        self.loss_G_His = self.criterionHistc(self.fake_B, self.real_B) /100.0
        self.alpha.append(self.loss_G_His)
        alpha = torch.Tensor(self.alpha)
        alpha = (alpha - alpha.mean())/alpha.std()
        alpha = torch.sigmoid(alpha)[-1]

        self.loss_G_HisGDL = self.loss_G_GDL * float(1.0+alpha)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_HisGDL

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.emaG.update()

# ************************************************  HMS_GANModel_GLGCM  ************************************************
class HMS_GANModel_GLGCM(BaseModel):
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_GAN', 'G_L1', 'G_HisGDL', 'glgcm_loss','D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.schedulers = [get_scheduler(optimizer, conf) for optimizer in self.optimizers]

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = define_G(
            conf.input_nc,
            conf.output_nc,
            conf.ngf,
            conf.which_model_netG,
            conf.vit_name,
            conf.fineSize,
            conf.pre_trained_path,
            conf.norm,
            not conf.no_dropout,
            conf.init_type,
            self.gpu_ids,
            pre_trained_trans=conf.pre_trained_transformer,
            pre_trained_resnet=conf.pre_trained_resnet,
        ).cuda()
        self.emaG = EMA(self.netG, 0.9999)
        self.emaG.register()

        if self.isTrain:
            self.netD = define_D(conf.in_channel + conf.out_channel, ndf=64, n_layers_D=3, norm='instance',init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionGDL = GDLoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionOutlineROI = OutlineROI().to(self.device)
            self.criterionGLGCM = GLGCM_Loss().to(self.device)
            self.criterionVGG = VGGLoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.fake_B_pool = ImagePool(50)
            self.alpha = [0.0]

    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.original_A = input['original_A' if task else 'original_B'].to(self.device)

        self.real_B = input['B' if task else 'A'].to(self.device)
        self.original_B = input['original_B' if task else 'original_A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']
        self.image_name = input['name']

    def forward(self):
        self.fake_B = self.netG(self.real_A.cuda()).cuda()

    def backward_D(self):
        fake_B = self.fake_B_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B, fake_B) * 10.0

        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_GP

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = 10.0 * self.criterionL1(self.fake_B, self.real_B).to(self.device)

        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0
        self.loss_G_His = self.criterionHistc(self.fake_B, self.real_B) /100.0
        self.alpha.append(self.loss_G_His)
        alpha = torch.Tensor(self.alpha)
        alpha = (alpha - alpha.mean())/alpha.std()
        alpha = torch.sigmoid(alpha)[-1]

        self.loss_G_HisGDL = self.loss_G_GDL * float(1.0+alpha)

        self.OutlineROI_real_B = self.criterionOutlineROI(self.real_B).to(self.device)
        self.OutlineROI_fake_B = self.criterionOutlineROI(self.fake_B).to(self.device)
        self.loss_glgcm_loss = self.criterionGLGCM(self.real_B, self.fake_B, self.OutlineROI_real_B,self.OutlineROI_fake_B).to(self.device)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_HisGDL + self.loss_glgcm_loss

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.emaG.update()

# ************************************************  HMS_GANModel_Diffusion  ************************************************
class HMS_GANModel_Diffusion(BaseModel):
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_total', 'G_GAN', 'G_L1', 'G_HisGDL', 'Diff_GAN', 'Diff_L1', 'Diff_HisGDL','D_real', 'D_fake','GP']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.schedulers = [get_scheduler(optimizer, conf) for optimizer in self.optimizers]
        self.save_file = conf.output_save_path

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = define_G(
            conf.input_nc,
            conf.output_nc,
            conf.ngf,
            conf.which_model_netG,
            conf.vit_name,
            conf.fineSize,
            conf.pre_trained_path,
            conf.norm,
            not conf.no_dropout,
            conf.init_type,
            self.gpu_ids,
            pre_trained_trans=conf.pre_trained_transformer,
            pre_trained_resnet=conf.pre_trained_resnet,
        ).cuda()
        # 扩散模型生成器
        self.Diff_G = define_Diff_G(conf).cuda()

        self.emaG = EMA(self.netG, 0.9999)
        self.emaG.register()

        if self.isTrain:
            self.netD = define_D(conf.in_channel + conf.out_channel, ndf=64, n_layers_D=3, norm='instance',init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionGDL = GDLoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionOutlineROI = OutlineROI().to(self.device)
            self.criterionGLGCM = GLGCM_Loss().to(self.device)
            self.criterionVGG = VGGLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.Diff_G.parameters()),lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.fake_B_pool = ImagePool(50)
            self.alpha = [0.0]
            self.diff_alpha = [0.0]

        # print("---------- Networks initialized -------------")
        # print_network(self.netG)

    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.original_A = input['original_A' if task else 'original_B'].to(self.device)

        self.real_B = input['B' if task else 'A'].to(self.device)
        self.original_B = input['original_B' if task else 'original_A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']
        self.image_name = input['name']

    def forward(self):
        self.fake_B = self.netG(self.real_A.cuda()).cuda()
        # 扩散模型生成的fake_B图像
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        alpha = Tensor(np.random.random((self.real_A.size(0), 1, 1, 1)))
        self.random_AB = (alpha * self.real_A + (1 - alpha) * self.fake_B).requires_grad_(True)
        self.Diff_fake_B = self.Diff_G(self.random_AB, self.original_B, self.image_name).to(self.device)

    def backward_D(self):
        fake_B = self.fake_B_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # 扩散模型生成的假图像
        Diffusion_fake_B = self.fake_B_pool.query(torch.cat((self.real_A, self.Diff_fake_B), 1).data)
        pred_Diffusion_fake = self.netD(Diffusion_fake_B.detach())
        self.loss_D_Diffusion_fake = self.criterionGAN(pred_Diffusion_fake, False)

        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B, fake_B) * 10.0

        self.loss_D = (self.loss_D_fake + self.loss_D_Diffusion_fake) * 0.5 + self.loss_D_real + self.loss_GP

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = 10.0 * self.criterionL1(self.fake_B, self.real_B).to(self.device)

        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0
        self.loss_G_His = self.criterionHistc(self.fake_B, self.real_B) /100.0
        self.alpha.append(self.loss_G_His)
        alpha = torch.Tensor(self.alpha)
        alpha = (alpha - alpha.mean())/alpha.std()
        alpha = torch.sigmoid(alpha)[-1]
        self.loss_G_HisGDL = self.loss_G_GDL * float(1.0+alpha)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_HisGDL

        fake_Diff_B = torch.cat((self.real_A, self.Diff_fake_B), 1)
        Diffusion_fake_B = self.netD(fake_Diff_B)

        self.loss_Diff_GAN = self.criterionGAN(Diffusion_fake_B, True)
        self.loss_Diff_L1 = 10.0 * self.criterionL1(self.Diff_fake_B, self.real_B).to(self.device)

        self.loss_Diff_GDL = self.criterionGDL(self.Diff_fake_B, self.real_B) * 10.0
        self.loss_Diff_His = self.criterionHistc(self.Diff_fake_B, self.real_B) / 100.0
        self.diff_alpha.append(self.loss_Diff_His)
        diff_alpha = torch.Tensor(self.diff_alpha)
        diff_alpha = (diff_alpha - diff_alpha.mean()) / diff_alpha.std()
        diff_alpha = torch.sigmoid(diff_alpha)[-1]
        self.loss_Diff_HisGDL = self.loss_Diff_GDL * float(1.0 + diff_alpha)
        self.loss_Diff = self.loss_Diff_GAN + self.loss_Diff_L1 + self.loss_Diff_HisGDL

        self.loss_G_total = self.loss_G + self.loss_Diff

        self.optimizer_G.zero_grad()
        self.loss_G_total.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.Diff_G.optimize_parameters(self.original_A, self.original_B, self.image_name)
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.emaG.update()

# *******************************************************  HMS_GANModel  *******************************************************
class HMS_GANModel(BaseModel):
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_total', 'G_GAN', 'G_L1', 'G_HisGDL', 'glgcm_loss', 'Diff_GAN', 'Diff_L1', 'Diff_HisGDL','D_real', 'D_fake','GP']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.schedulers = [get_scheduler(optimizer, conf) for optimizer in self.optimizers]
        self.save_file = conf.output_save_path

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = define_G(
            conf.input_nc,
            conf.output_nc,
            conf.ngf,
            conf.which_model_netG,
            conf.vit_name,
            conf.fineSize,
            conf.pre_trained_path,
            conf.norm,
            not conf.no_dropout,
            conf.init_type,
            self.gpu_ids,
            pre_trained_trans=conf.pre_trained_transformer,
            pre_trained_resnet=conf.pre_trained_resnet,
        ).cuda()
        # 扩散模型生成器
        self.Diff_G = define_Diff_G(conf).cuda()
        self.emaG = EMA(self.netG, 0.9999)
        self.emaG.register()
        if self.isTrain:
            self.netD = define_D(conf.in_channel + conf.out_channel, ndf=64, n_layers_D=3, norm='instance',init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.criterionGAN = GANLoss().to(self.device)
            # L1 损失函数
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            # 梯度差异损失函数
            self.criterionGDL = GDLoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            # 均方误差（MSE）损失函数
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionOutlineROI = OutlineROI().to(self.device)
            self.criterionGLGCM = GLGCM_Loss().to(self.device)
            self.criterionVGG = VGGLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.Diff_G.parameters()),lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.fake_B_pool = ImagePool(50)
            self.alpha = [0.0]
            self.diff_alpha = [0.0]

    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.original_A = input['original_A' if task else 'original_B'].to(self.device)

        self.real_B = input['B' if task else 'A'].to(self.device)
        self.original_B = input['original_B' if task else 'original_A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']
        self.image_name = input['name']

    def forward(self):
        self.fake_B = self.netG(self.real_A.cuda()).cuda()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        alpha = Tensor(np.random.random((self.real_A.size(0), 1, 1, 1)))
        self.random_AB = (alpha * self.real_A + (1 - alpha) * self.fake_B).requires_grad_(True)
        self.Diff_fake_B = self.Diff_G(self.random_AB, self.original_B, self.image_name).to(self.device)

    def backward_D(self):
        fake_B = self.fake_B_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # 扩散模型生成的假图像
        Diffusion_fake_B = self.fake_B_pool.query(torch.cat((self.real_A, self.Diff_fake_B), 1).data)
        pred_Diffusion_fake = self.netD(Diffusion_fake_B.detach())
        self.loss_D_Diffusion_fake = self.criterionGAN(pred_Diffusion_fake, False)

        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B, fake_B) * 10.0

        self.loss_D = (self.loss_D_fake + self.loss_D_Diffusion_fake) * 0.5 + self.loss_D_real + self.loss_GP

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = 10.0 * self.criterionL1(self.fake_B, self.real_B).to(self.device)

        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0
        self.loss_G_His = self.criterionHistc(self.fake_B, self.real_B) /100.0
        self.alpha.append(self.loss_G_His)
        alpha = torch.Tensor(self.alpha)
        alpha = (alpha - alpha.mean())/alpha.std()
        alpha = torch.sigmoid(alpha)[-1]

        self.loss_G_HisGDL = self.loss_G_GDL * float(1.0+alpha)

        self.OutlineROI_real_B = self.criterionOutlineROI(self.real_B).to(self.device)
        self.OutlineROI_fake_B = self.criterionOutlineROI(self.fake_B).to(self.device)
        self.loss_glgcm_loss = self.criterionGLGCM(self.real_B, self.fake_B, self.OutlineROI_real_B,self.OutlineROI_fake_B).to(self.device)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_HisGDL + self.loss_glgcm_loss

        fake_Diff_B = torch.cat((self.real_A, self.Diff_fake_B), 1)
        Diffusion_fake_B = self.netD(fake_Diff_B)

        self.loss_Diff_GAN = self.criterionGAN(Diffusion_fake_B, True)
        self.loss_Diff_L1 = 10.0 * self.criterionL1(self.Diff_fake_B, self.real_B).to(self.device)

        self.loss_Diff_GDL = self.criterionGDL(self.Diff_fake_B, self.real_B) * 10.0
        self.loss_Diff_His = self.criterionHistc(self.Diff_fake_B, self.real_B) / 100.0
        self.diff_alpha.append(self.loss_Diff_His)
        diff_alpha = torch.Tensor(self.diff_alpha)
        diff_alpha = (diff_alpha - diff_alpha.mean()) / diff_alpha.std()
        diff_alpha = torch.sigmoid(diff_alpha)[-1]
        self.loss_Diff_HisGDL = self.loss_Diff_GDL * float(1.0 + diff_alpha)

        self.loss_Diff = self.loss_Diff_GAN + self.loss_Diff_L1 + self.loss_Diff_HisGDL

        self.loss_G_total = self.loss_G + self.loss_Diff * 0.5

        self.optimizer_G.zero_grad()
        self.loss_G_total.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.Diff_G.optimize_parameters(self.original_A, self.original_B, self.image_name)
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.emaG.update()
