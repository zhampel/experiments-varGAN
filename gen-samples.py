from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    from scipy.optimize import curve_fit

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import ScalarFormatter

    import tables

    import pandas as pd
    
    import torch
    
    from vargan.definitions import DATASETS_DIR
    from vargan.utils import sample_z
except ImportError as e:
    print(e)
    raise ImportError

    
# Marker/Line Options
colors = ["blue", "red", "green", "black"]
colorsmall = ["b", "r", "g", "k"]
styles = ["-", "--", "-."]


def main():
    global args
    parser = argparse.ArgumentParser(description="Gaussian dataset generation script")
    parser.add_argument("-b", "--n_batches", dest="n_batches", default=100, type=int, help="Number of batches")
    parser.add_argument("-n", "--n_samples", dest="n_samples", default=1024, type=int, help="Number of samples")
    parser.add_argument("-d", "--dim_list", dest="dim_list", nargs='+', default=[1, 2, 3, 10, 100], type=int, help="Number of samples")
    args = parser.parse_args()

    # Make directory structure for this run
    data_dir = os.path.join(DATASETS_DIR)
    os.makedirs(data_dir, exist_ok=True)
    print('\nDatasets to be saved in directory %s\n'%(data_dir))
    atom = tables.Float64Atom()

    # Number of samples to take
    n_samples = args.n_samples
    n_batches = args.n_batches
    n_total = int(n_samples * n_batches)
    
    # List of dimensions to test
    dim_list = args.dim_list
    n_dims = len(dim_list) 
    
    # Prepare component histograms
    xedges = np.arange(-4, 4, 0.25)
    xcents = (xedges[1:]-xedges[:-1])/2 + xedges[0:-1]

    print("Generating data for dimensions: ", dim_list)
    print("Each dataset will have %i samples."%n_total)

    # Define histograms for saving generated data
    redges = np.arange(0, 50, 0.25)
    rcents = (redges[1:]-redges[:-1])/2 + redges[0:-1]
   
    # Loop through dimensions list
    for idx, dim in enumerate(dim_list):
    
        # Initialize histogram for dimensionality
        xhist, _ = np.histogram([], bins=xedges)
        xhist_list = [xhist] * dim
        pdims = np.ceil(np.sqrt(dim))

        # Prepare file, earray to save generated data
        data_file_name = '%s/data_dim%i.h5'%(data_dir, dim)
        data_file = tables.open_file(data_file_name, mode='w')
        atom = tables.Float64Atom()
        array_c = data_file.create_earray(data_file.root, 'data', atom, (0, dim))

        # Run through number of batches, getting n_samples each
        for ibatch in range(n_batches):
            # Random set of n_samples
            z = sample_z(samples=n_samples, dims=dim, mu=0.0, sigma=1.0)
            z_numpy = z.cpu().data.numpy()

            # Add norm entry to histogram
            for idim in range(dim):
                xhist_list[idim] += np.histogram(z_numpy[:, idim], bins=xedges)[0]

            # Add samples dataset
            array_c.append(z_numpy)
 
        # Close dataset file
        data_file.close()

        ## Generate figures
        figname = '%s/hists_dim%i.png'%(data_dir, dim)
        fig = plt.figure(figsize=(18,12))
        mpl.rc("font", family="serif")
        for idim in range(dim):
            # Access & normalize histogram for each component
            xhist = xhist_list[idim]
            xhist = xhist / np.sum(xhist)
            
            iax = fig.add_subplot(pdims, pdims, idim + 1)
            iax.step(xcents, xhist, linewidth=1.5, c='r')
            iax.grid()
            iax.set_xlim(-4, 4)
            iax.set_ylim(0.0, 0.25)
            plt.axis('off')
            
        plt.tight_layout()
        fig.savefig('%s'%figname)
   
    
if __name__ == "__main__":
    main()
