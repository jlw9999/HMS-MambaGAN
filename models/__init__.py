from .HMSMambaGAN import HMS_GANModel_hms,HMS_GANModel_GLGCM,HMS_GANModel_Diffusion,HMS_GANModel

def create_model(opt):
    model = None
    print(opt.model)
