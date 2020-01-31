import numpy as np
import sys
from sys import stdout
import time
import theano
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from model_wlat_smep_mod import Network as Network_SMEP_Mod

def create_hyp_param_combination(hidden_sizes=[500, 500, 500],
	n_epochs=200,
	batch_size=20,
	n_it_neg=500,
	n_it_pos=8,
	epsilon=np.float32(.5),
	beta=np.float32(1.),
	alphas_fwd=[np.float32(.128), np.float32(.032), np.float32(.008), np.float32(.002)],
	alphas_lat=[np.float32(0.75), np.float32(0.375), np.float32(0.09375)],
    alphas_unsuper=[np.float32(0.01), np.float32(0.01)],
	beta_reg_bool=False,
	alpha_tdep_type='constant',
	dataset="mnist",
	variant="normal",
    super_thr=np.float32(0.5)
):
	hp_dict  = {
	"hidden_sizes": hidden_sizes,
	"n_epochs": n_epochs,
	"batch_size": batch_size,
	"n_it_neg": n_it_neg,
	"n_it_pos": n_it_pos,
	"epsilon": epsilon,
	"beta": beta,
	"alphas_fwd": alphas_fwd,
	"alphas_lat": alphas_lat,
    "alphas_unsuper": alphas_unsuper,
	"beta_reg_bool": beta_reg_bool,
	"alpha_tdep_type": alpha_tdep_type,
	"dataset": dataset,
	"variant": variant,
    "super_thr":super_thr
    }
	return hp_dict


def train_net(net):
    print "Learning rule: SMEP"
    path = net.path
    hidden_sizes = net.hyperparameters["hidden_sizes"]
    n_epochs = net.hyperparameters["n_epochs"]
    batch_size = net.hyperparameters["batch_size"]
    n_it_neg = net.hyperparameters["n_it_neg"]
    n_it_pos = net.hyperparameters["n_it_pos"]
    epsilon = net.hyperparameters["epsilon"]
    beta = net.hyperparameters["beta"]
    alphas_fwd = net.hyperparameters["alphas_fwd"]
    alphas_lat = net.hyperparameters["alphas_lat"]
    beta_regularizer_boolean = net.hyperparameters["beta_reg_bool"]
    alpha_tdep_type = net.hyperparameters["alpha_tdep_type"]
    mod = net.hyperparameters["variant"]

    print "name = %s" % (path)
    print "architecture = 784-" + "-".join([str(n) for n in hidden_sizes]) + "-10"
    print "number of epochs = %i" % (n_epochs)
    print "batch_size = %i" % (batch_size)
    print "n_it_neg = %i" % (n_it_neg)
    print "n_it_pos = %i" % (n_it_pos)
    print "epsilon = %.1f" % (epsilon)
    print "beta = %.1f" % (beta)
    print "learning rates forward: " + " ".join(
        ["alpha_W%i=%.3f" % (k + 1, alpha) for k, alpha in enumerate(alphas_fwd)]) + "\n"
    print "learning rates lateral: " + " ".join(
        ["alpha_L%i=%.3f" % (k + 1, alpha) for k, alpha in enumerate(alphas_lat)]) + "\n"
    print "beta sign alternating: ", beta_regularizer_boolean
    print "alpha tdep: ", alpha_tdep_type
    print "variant: ", mod
    print "Checking non linearity used: ", net.rho(-3.0).eval()

    n_batches_train = 50000 / batch_size
    n_batches_valid = 1000 / batch_size

    # DEFINE THE SUPERVISED FLAG FOR EACH BATCH

    start_time = time.clock()
    start_epoch = len(net.training_curves['training error'])
    print 'Continuing training post Epoch Number', start_epoch

    for epoch in range(n_epochs):
        ### TRAINING ###

        # CUMULATIVE SUM OF TRAINING ENERGY, TRAINING COST AND TRAINING ERROR
        measures_sum_val = np.zeros(3, dtype=theano.config.floatX)
        measures_avg_val = np.zeros(3, dtype=theano.config.floatX)



        alphas_arg = alphas_fwd + alphas_lat
        print 'Learning Rate for epoch: ', alphas_arg

        ### VALIDATION ###
        # CUMULATIVE SUM OF VALIDATION ENERGY, VALIDATION COST AND VALIDATION ERROR

        for index in xrange(n_batches_valid):
            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(n_batches_train + index)

            # FREE PHASE
            net.free_phase(n_it_neg, epsilon)

            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum_val = measures_sum_val + np.array(measures)
            layers = [lay.eval() for lay in net.layers]
            y_predict = np.argmax(layers[-1], axis=1)

            # reorder based on the size
            ordered_inx = np.argsort(y_predict)
            ordered_layers = [lay[ordered_inx, :] for lay in layers]


            if index == (n_batches_valid - 1):
                measures_avg_val = measures_sum_val / (index + 1)
                measures_avg_val[
                    -1] *= 100.  # measures_avg[-1] corresponds to the error rate, which we want in percentage
                stdout.write("\r   valid-%5i E=%.1f C=%.5f error=%.2f%%" % (
                (index + 1) * batch_size, measures_avg_val[0], measures_avg_val[1], measures_avg_val[2]))
                stdout.flush()

        stdout.write("\n")

        net.training_curves["validation error"].append(measures_avg_val[-1])


        duration = (time.clock() - start_time) / 60.
        print("   duration=%.1f min" % (duration))

        # SAVE THE PARAMETERS OF THE NETWORK AT THE END OF THE EPOCH
        #net.save_params()

        # calculate and Plot the similarity matrices
        # calculate the similarity matrices
        SM = [np.matmul(lay,np.transpose(lay)) for lay in ordered_layers]
        pp = PdfPages('mnist_SMEP_SM.pdf')
        fontsize = 20
        plt.figure(figsize=(10, 9))
        # for i in range(len(SM)):
        #     plt.figure(i)
        #     ax0 = plt.subplot(2, 2, i)
        #     im0 = plt.imshow(SM[i])
        #     plt.colorbar(im0, use_gridspec=True)
        #     ax0.set_title('x', fontsize=fontsize)
        # plt.tight_layout()  # tigth layout
        # pp.savefig()
        # pp.close()

        print('done!')
        plt.figure(1)
        ax0 = plt.subplot(2, 2, 1)
        im0 = plt.imshow(SM[0])
        plt.colorbar(im0, use_gridspec=True)
        #ax0.set_title('x', fontsize=fontsize)

        ax1 = plt.subplot(2, 2, 2)
        #ax1.set_title('h_conv1', fontsize=fontsize)
        im1 = plt.imshow(SM[1])
        plt.colorbar(im1, use_gridspec=True)

        ax2 = plt.subplot(2, 2, 3)
        im2 = plt.imshow(SM[2])
        plt.colorbar(im2, use_gridspec=True)
        #ax2.set_title('h_conv2', fontsize=fontsize)

        # ax3 = plt.subplot(2, 2, 4)
        # im3 = plt.imshow(SM3)
        # plt.colorbar(im3, use_gridspec=True)
        # ax3.set_title('fc', fontsize=fontsize)

        plt.tight_layout()  # tigth layout
        pp.savefig()
        pp.close()


if __name__=='__main__':
	#fract_str = sys.argv[3]
	#PART1: Code for 1 HL, single run, specifying hyperparams
	dirname='Results/' #CHANGE DIRECTORY CHOICE HERE, .save file will be stored in this directory
	#name = dirname + 'smep'+ fract_str
	name = dirname + 'smep'
	hpd1 = create_hyp_param_combination(hidden_sizes = [500],
		n_epochs=1,
		batch_size=1000,
		n_it_neg=20,
		n_it_pos=4,
		epsilon=np.float32(.5),
		beta=np.float32(1.),
		beta_reg_bool=False,
		alpha_tdep_type='constant',
		dataset="mnist",
		variant="normal",
		alphas_fwd= [np.float32(0.5), np.float32(0.375)], #0.5, 0.375, 0.281, 0.211
        alphas_unsuper=[np.float32(0.5),np.float32(0.375)/2],  #Update for unsupervised
		alphas_lat=[np.float32(0.01)], #CHANGE ALPHA LATERAL HERR, 0.01
        super_thr = np.float32(1),  # The ratio of superivsed samples, in the range of [0,1]
	)
	train_net(Network_SMEP_Mod(name, hpd1))