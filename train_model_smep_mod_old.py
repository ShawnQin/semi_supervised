import numpy as np
import sys
from sys import stdout
import time
import theano
import theano.tensor as T

'''
SMEP
Code for 
- beta signed or positive
- different variants (clipping lower bound 0 vs -1)
- alpha time dependence cases

Restrictions
- Dataset: MNIST only
'''



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
	mod=net.hyperparameters["variant"]
	dataset=net.hyperparameters["dataset"]
	
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
	n_batches_valid = 10000 / batch_size

	start_time = time.clock()
	start_epoch=len(net.training_curves['training error'])
	print 'Continuing training post Epoch Number', start_epoch
	for epoch in range(n_epochs):

		### TRAINING ###

		# CUMULATIVE SUM OF TRAINING ENERGY, TRAINING COST AND TRAINING ERROR
		measures_sum_tr=np.zeros(7, dtype=theano.config.floatX)
		measures_avg_tr=np.zeros(7, dtype=theano.config.floatX)
		measures_sum_val = np.zeros(3, dtype=theano.config.floatX) 
		measures_avg_val=np.zeros(3, dtype=theano.config.floatX)
		gW = [0.] * len(alphas_fwd) #fix this?
		gL = [0.] * len(alphas_lat)
		#gLnorm = [0.] * len(alphas_lat)
		

		#Learning rate time dependence
		if alpha_tdep_type=='alphatd1': #EXP DECR
			alphas_lat = [np.float32(alpha /(1.0+epoch)) for alpha in alphas_lat]
		if alpha_tdep_type=='alphatd2':#EXP DECR
			alphas_lat = [np.float32(alpha /(1.0+(0.5*epoch))) for alpha in alphas_lat]
		if alpha_tdep_type=='alphalinear':
			alphas_lat = [np.float32(alpha /(1.0+epoch)) for alpha in net.hyperparameters["alphas_lat"]]
		if alpha_tdep_type=='cont_alphasmall':
			alphas_lat =list(np.array([0.005, 0.001, 0.0005], dtype=np.float32))
		if alpha_tdep_type=='alpha_segmented':
			if epoch<40:
				alphas_lat = net.hyperparameters["alphas_lat"]
			elif (epoch>=40) and (epoch<60):
				alphas_lat = list(np.array(net.hyperparameters["alphas_lat"])/10.0) 
			else:
				alphas_lat = list(np.array(net.hyperparameters["alphas_lat"])/100.0)
		if alpha_tdep_type=='alpha_segmented_old_40':
			if epoch<40:
				alphas_lat = net.hyperparameters["alphas_lat"]
			elif (epoch>=40) and (epoch<60):
				alphas_lat = list(np.array([0.05, 0.01, 0.005], dtype=theano.config.floatX))
			else:
				alphas_lat = list(np.array([0.005, 0.001, 0.0005], dtype=theano.config.floatX))	
		if alpha_tdep_type=='alpha_segmented_repr':
			if epoch<70:
				alphas_lat = net.hyperparameters["alphas_lat"]
			else:
				alphas_lat = list(np.array([0.05, 0.01, 0.005], dtype=theano.config.floatX))
		alphas_arg = alphas_fwd + alphas_lat
		print 'Learning Rate for epoch: ', alphas_arg
		for index in xrange(n_batches_train):
			# CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
			net.change_mini_batch_index(index)

			# FREE PHASE
			net.free_phase(n_it_neg, epsilon)

			np.mean(net.weights['lat'][0].get_value())
			# MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
			measures_verb_tr = net.measure_verbose() #E, C, Error, sq_n, lin_term, quad_term, quad_lat
			measures_sum_tr = measures_sum_tr + np.array(measures_verb_tr)

			if index==(n_batches_train-1):
				measures_avg_tr = measures_sum_tr/np.float32(index+1)
				measures_avg_tr[2] *= 100.  # measures_avg[-1] corresponds to the error rate, which we want in percentage
				stdout.write("\r%2i-train-%5i E=%.1f C=%.5f SqNorm=%.2f LinTerm=%.2f QuadTerm=%.2f QuadLatTerm=%.2f error=%.3f%%" % (
				(epoch+start_epoch), (index + 1) * batch_size, measures_avg_tr[0], measures_avg_tr[1], measures_avg_tr[3], measures_avg_tr[4], measures_avg_tr[5], measures_avg_tr[6], measures_avg_tr[2]))
				stdout.flush()

			# WEAKLY CLAMPED PHASE
			#Beta positive or regularized
			if beta_regularizer_boolean==True:
				sign = 2 * np.random.randint(0, 2) - 1	#alphas_random sign +1 or -1
				beta = np.float32(sign*beta)
			else:
				beta = np.float32(beta)  # choose the sign of beta at random
			Delta_log_all = net.weakly_clamped_phase(n_it_pos, epsilon, beta, *alphas_arg)
			#wdot_listarr = net.weakly_clamped_phase(n_it_pos, epsilon, beta, *alphas_arg)
			#Delta_logW, Delta_logL = Delta_log_all[:len(alphas_fwd)], Delta_log_all[len(alphas_fwd):len(alphas_fwd)+len(alphas_lat)]
			#Sdl_w, Sdl_l = wdot_listarr[len(alphas_fwd)+len(alphas_lat):(2*len(alphas_fwd))+len(alphas_lat)], wdot_listarr[(2*len(alphas_fwd))+len(alphas_lat): 2*(len(alphas_fwd)+len(alphas_lat))]
			#print ('Batch-wise change (Dlw1, Dlw2, Dll): ', sign, Sdl_w, Sdl_l)
			#print (4)
			'''
			beta = np.float32(sign * beta)	# choose the sign of beta at random
			alphas_arg = alphas_fwd+alphas_lat
			pooled_outputs = net.weakly_clamped_phase(n_it_pos, epsilon, beta, *alphas_arg) #changd to examine
			Delta_log_all, biases_fwd_new, weights_fwd_new, weights_lat_new = np.array(pooled_outputs[0]), np.array(pooled_outputs[1]), np.array(pooled_outputs[2]), np.array(pooled_outputs[3])
			Delta_logW, Delta_logL = Delta_log_all[:len(alphas_fwd)], Delta_log_all[len(alphas_fwd):] #changd to examine'''
			#print ('Batch-wise change (Dlw1, Dlw2, Dll): ', Delta_logW, Delta_logL)

			Delta_logW, Delta_logL = Delta_log_all[:len(alphas_fwd)], Delta_log_all[len(alphas_fwd):len(alphas_fwd)+len(alphas_lat)]
			#Delta_logW, Delta_logL, Delta_logLnorm = Delta_log_all[:len(alphas_fwd)], Delta_log_all[len(alphas_fwd):len(alphas_fwd)+len(alphas_lat)], Delta_log_all[len(alphas_fwd)+len(alphas_lat):]
			gW = [gW1 + Delta_logW1 for gW1, Delta_logW1 in zip(gW, Delta_logW)]
			gL = [gL1 + Delta_logL1 for gL1, Delta_logL1 in zip(gL, Delta_logL)]
			#gLnorm = [gL1norm + Delta_logL1norm for gL1norm, Delta_logL1norm in zip(gLnorm, Delta_logLnorm)]
		stdout.write("\n")
		dlogW = [100. * gW1 / n_batches_train for gW1 in gW]
		dlogL = [100. * gL1 / n_batches_train for gL1 in gL]
		#dlogLnorm = [100. * gL1norm / n_batches_train for gL1norm in gLnorm]
		print "   " + " ".join(["dlogW%i=%.3f%%" % (k + 1, dlogW1) for k, dlogW1 in enumerate(dlogW)])
		print "   " + " ".join(["dlogL%i=%.3f%%" % (k + 1, dlogL1) for k, dlogL1 in enumerate(dlogL)])
		#print "   " + " ".join(["dlogLnorm%i=%.3f%%" % (k + 1, dlogL1norm) for k, dlogL1norm in enumerate(dlogLnorm)])
		# add print delta w / w and delta l/l in here somewhere

		net.training_curves["training error"].append(measures_avg_tr[2])

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

			if index==(n_batches_valid-1):
				measures_avg_val = measures_sum_val / (index + 1)
				measures_avg_val[-1] *= 100.  # measures_avg[-1] corresponds to the error rate, which we want in percentage
				stdout.write("\r   valid-%5i E=%.1f C=%.5f error=%.2f%%" % ((index + 1) * batch_size, measures_avg_val[0], measures_avg_val[1], measures_avg_val[2]))
				stdout.flush()

		stdout.write("\n")

		net.training_curves["validation error"].append(measures_avg_val[-1])

		duration = (time.clock() - start_time) / 60.
		print("   duration=%.1f min" % (duration))

		# SAVE THE PARAMETERS OF THE NETWORK AT THE END OF THE EPOCH
		net.save_params()

# HYPERPARAMETERS FOR NETWORKS WITH LATERAL CONNECTIONS

net1_lat_gradadd_betapos_divabs = "net1_lat_gradadd_betapos_divabs", {
		"hidden_sizes": [500],
		"n_epochs": 25,
		"batch_size": 20,
		"n_it_neg": 20,
		"n_it_pos": 4,
		"epsilon": np.float32(.5),
		"beta"	 : np.float32(1.),
		"alphas_fwd": [np.float32(0.1), np.float32(.05)],
		"alphas_lat": [np.float32(0.03)]
	}

net2_lat = "net2_lat", {
	"hidden_sizes": [500, 500],
	"n_epochs": 40,
	"batch_size": 20,
	"n_it_neg": 150,
	"n_it_pos": 6,
	"epsilon": np.float32(.5),
	"beta": np.float32(1.),
	"alphas_fwd": [np.float32(.4), np.float32(.1), np.float32(.01)],
	"alphas_lat": [ np.float32(.4), np.float32(.1)]
}

net3_lat = "net3_lat", {
	"hidden_sizes": [500, 500, 500],
	"n_epochs": 500,
	"batch_size": 20,
	"n_it_neg": 500,
	"n_it_pos": 8,
	"epsilon": np.float32(.5),
	"beta": np.float32(1.),
	"alphas_fwd": [np.float32(.128), np.float32(.032), np.float32(.008), np.float32(.002)],
	"alphas_lat": [np.float32(.128), np.float32(.032), np.float32(.008)]
}

if __name__ == "__main__":
	# TRAIN A NETWORK WITH 1 HIDDEN LAYER
	#train_net(Network(*net1_lat_gradadd_betapos_divabs))
	print 'main of trainmodel'
