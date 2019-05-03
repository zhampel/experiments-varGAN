from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from scipy import stats

    import tables

    import pandas as pd
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from itertools import chain as ichain

    from gaussgan.definitions import DATASETS_DIR, RUNS_DIR
    from gaussgan.datasets import get_dataloader, GaussDataset
    from gaussgan.models import Generator, Discriminator
    from gaussgan.utils import tlog, save_model, enorm, calc_gradient_penalty, sample_z, sample_zu
    from gaussgan.plots import plot_train_loss, compare_histograms
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="GAN training script")
    parser.add_argument("-d", "--dim", dest="dimensions", default=1, type=int, help="Number of dimensions")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=20, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    args = parser.parse_args()

    dim = args.dimensions
    run_name = 'dim%i'%dim

    # Make directory structure for this run
    run_dir = os.path.join(RUNS_DIR, run_name)
    data_dir = os.path.join(DATASETS_DIR)
    samples_dir = os.path.join(run_dir, 'samples')
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))
   
    # Access saved dataset
    data_file_name = '%s/data_dim%i.h5'%(data_dir, dim)
    dataset = GaussDataset(file_name=data_file_name)
    print("Getting dataset from %s"%data_file_name)
    print("Dataset size: ", dataset.__len__())

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    # Latent space info
    latent_dim = 30
    latent_sigma = 0.01
   
    # Wasserstein metric flag
    #wass_metric = False
    wass_metric = True
    
    x_shape = dim
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    generator = Generator(latent_dim, x_shape)
    discriminator = Discriminator(dim=dim, wass_metric=wass_metric)
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Configure training data loader
    dataloader = get_dataloader(dataset=dataset, batch_size=batch_size)

    # Test dataset
    n_test_samples = 1000
    test_data = sample_z(samples=n_test_samples, dims=dim, mu=0.0, sigma=1.0)
    r_test = enorm(test_data)

    # Prepare test set component histograms
    test_hist_list = [None] * dim
    xedges = np.arange(-4, 4, 0.5)
    xcents = (xedges[1:]-xedges[:-1])/2 + xedges[0:-1]
    pdims = np.ceil(np.sqrt(dim))
    test_data_numpy = test_data.cpu().data.numpy()

    for idim in range(dim):
        # Distribution for each dimension component
        xhist = np.histogram(test_data_numpy[:, idim], bins=xedges)[0]
        xhist = xhist / np.sum(xhist)
        test_hist_list[idim] = xhist

    # Theoretical dataset
    chi2_rng = np.random.chisquare(dim, n_test_samples)
    chi2_sqrt = np.sqrt(chi2_rng)

    # K-S Test
    dval, pval = stats.ks_2samp(r_test, chi2_sqrt)
    print("Comparing theoretical chi2 dist (sqrt) to sampled distribution:")
    print("P-Value: %.04f\tDist-Value: %.04f"%(dval, pval))

    # Bin distributions
    redges = np.linspace(0, 1.2*int(np.max(r_test)), 20)
    rcents = (redges[1:]-redges[:-1])/2 + redges[0:-1]
    test_hist, _ = np.histogram(r_test, bins=redges)
    chi_hist, _ = np.histogram(chi2_sqrt, bins=redges)
    test_hist = test_hist / float(n_test_samples)
    chi_hist = chi_hist / float(n_test_samples)
   
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

    # ----------
    #  Training
    # ----------
    g_l = []
    d_l = []
    dval_i = []
    pval_i = []
    
    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
        for i, samples in enumerate(dataloader):
           
            # Ensure generator is trainable
            generator.train()
            # Zero gradients for models
            generator.zero_grad()
            discriminator.zero_grad()
            
            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # -----------------
            #  Train Generator 
            # -----------------
            
            optimizer_G.zero_grad()
            
            # Sample random latent variables
            #z_latent = sample_z(samples=real_samples.size()[0], dims=latent_dim, mu=0.0, sigma=latent_sigma)
            z_latent = sample_zu(samples=real_samples.size()[0], dims=latent_dim)

            # Generate a batch of samples
            gen_samples = generator(z_latent)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_samples)
            D_real = discriminator(real_samples)
            
            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % n_skip_iter == 0):
    
                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    g_loss = torch.mean(D_gen)
                else:
                    # Vanilla GAN loss
                    g_loss = -torch.mean(tlog(D_gen))
    
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_samples, gen_samples)

                # Wasserstein GAN loss w/gradient penalty
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
                
            else:
                # Vanilla GAN loss
                d_loss = -torch.mean(tlog(D_real) - tlog(1 - D_gen))
    
            d_loss.backward()
            optimizer_D.step()


        # Save training losses
        d_l.append(d_loss.item())
        g_l.append(g_loss.item())
   
        # Generator in eval mode
        generator.eval()

        # Set number of examples for cycle calcs
        n_samp = 1000 


        # Generate sample instances
        z_samp = sample_zu(samples=n_samp, dims=latent_dim)
        #z_samp = sample_z(samples=n_samp, dims=latent_dim, mu=0.0, sigma=latent_sigma)
        gen_samples_samp = generator(z_samp)


        # Compare true/gen distributions for each component
        gen_data_numpy = gen_samples_samp.cpu().data.numpy()

        ks_d_list = []
        ks_p_list = []

        figname = '%s/comp_hist_epoch%05i.png'%(samples_dir, epoch)
        fig = plt.figure(figsize=(18,12))
        mpl.rc("font", family="serif")
        for idim in range(dim):
            # Initialize histogram for each dimension component
            xhist = np.histogram(gen_data_numpy[:, idim], bins=xedges)[0]
            xhist = xhist / np.sum(xhist)
            
            iax = fig.add_subplot(pdims, pdims, idim + 1)
            iax.step(xcents, test_hist_list[idim], linewidth=1.5, c='k')
            iax.step(xcents, xhist, linewidth=1.5, c='r')
            iax.grid()
            iax.set_xlim(-4, 4)
            iax.set_ylim(0.0, 0.5)
            #iax.set_ylim(0.01, 0.75)
            #plt.yscale('log')
            plt.axis('off')
            
            dval, pval = stats.ks_2samp(gen_data_numpy[:, idim], test_data_numpy[:, idim])
            ks_d_list.append(dval)
            ks_p_list.append(pval)
        
        plt.tight_layout()
        fig.savefig('%s'%figname)
   
        # Save results of KS test to figure
        figname = '%s/ks_epoch%05i.png'%(samples_dir, epoch)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        axd = fig.add_subplot(111)
        # D-Values
        axd.step(np.arange(0, dim, 1), ks_d_list)
        axd.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$')
        axd.set_xlabel(r'Vector Component')
        axd.set_title(r'Results of KS Test for Each Component')
        # P-Values
        axp = axd.twinx()
        axp.step(np.arange(0, dim, 1), ks_p_list, c='r')
        axp.set_ylabel(r'$\mathrm{KS}_{\mathrm{p}}$')
        axp.tick_params('y', colors='r')
        fig.tight_layout()
        fig.savefig(figname)


        # Euclidean norm calc and comparison
        r_gen_samps = enorm(gen_samples_samp)
        # Bin samples into normalized histogram
        gen_hist, _ = np.histogram(r_gen_samps, bins=redges)
        gen_hist = gen_hist / float(n_samp)

        # Plot norm distributions
        figname = '%s/rhist_epoch%05i.png'%(samples_dir, epoch)
        compare_histograms(hist_list=[test_hist, gen_hist],
                           centers=[rcents, rcents],
                           labels=['Parent', 'Generated'],
                           ylims=[0, 1.0, False],
                           figname=figname)
      
        # K-S test test btw test data and generated samples (r distribution)
        dval, pval = stats.ks_2samp(r_test, r_gen_samps)
        
        dval_i.append(dval)
        pval_i.append(pval)

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [G: %f] [p-val: %.02e]" % (epoch, 
                                                                   n_epochs, 
                                                                   d_loss.item(),
                                                                   g_loss.item(),
                                                                   pval)
              )
        

    # Save training results
    train_df = pd.DataFrame({
                             'dim' : dim,
                             'n_epochs' : n_epochs,
                             'learning_rate' : lr,
                             'beta_1' : b1,
                             'beta_2' : b2,
                             'weight_decay' : decay,
                             'n_skip_iter' : n_skip_iter,
                             'latent_dim' : latent_dim,
                             'latent_sigma' : latent_sigma,
                             'wass_metric' : wass_metric,
                             'gen_loss' : ['G', g_l],
                             'disc_loss' : ['D', d_l],
                             'pvalues' : ['pvals', pval_i],
                             'dvalues' : ['dvals', dval_i],
                            })

    train_df.to_csv('%s/training_details.csv'%(run_dir))


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_loss', 'disc_loss'],
                    figname='%s/training_model_losses.png'%(run_dir)
                    )

    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['pvalues'],
                    figname='%s/training_pvalues.png'%(run_dir)
                    )


    # Save current state of trained models
    model_list = [discriminator, generator]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()