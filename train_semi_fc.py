import numpy as np
import sys

#from train_model_wlat_smep import train_net as train_net_lat_smep
#from model_wlat_smep import Network as Network_Lateral_SMEP
#from train_model_wlat_ep_cif import train_net as train_net_lat_cif
#from model_wlat_ep_cif import Network as Network_Lateral_Cif
from train_model_smep_mod import train_net as train_net_smep_mod
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

def run_multiple(dirname, L=3, rule='eplat', guess=0, dataset="mnist", exp_type="normal"):
	#Check continuing run
	name = dirname+rule+'_{}'.format(guess)

	if dataset=="cifar10":
		name="CIFAR10/"+name+'_cif' #Changing parent directory, cif
	if ('Continuing' in dirname) or ('cont' in dirname):
		state_file=np.load(name+'.save', allow_pickle=True)
		tr_err, val_err=np.array(state_file[4]['training error']), np.array(state_file[4]['validation error'])
		print 'Initial Network Errors: ', tr_err[-1], val_err[-1], 'Continuing training post epochs: ', len(tr_err)


	if L==3 and rule=='eplat':
		if guess==0:
			alphas_w = np.array([0.128, 0.032, 0.008, 0.002], dtype=np.float32)

		if guess==1:
			alphas_w=np.zeros(L+1)
			tau_w=0.25
			alphas_w[0]=0.5
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w
		if guess==2:
			alphas_w=np.zeros(L+1)
			tau_w=0.5
			alphas_w[0]=0.5
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w
		
		if (guess==3) or (guess==4):	
			alphas_w=np.zeros(L+1)
			tau_w=0.75
			alphas_w[0]=0.5
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w
		if (guess==4):
			tau_l=np.float32(1.2)
		else:
			tau_l=np.float32(1.5)
		alphas_l=alphas_w[:-1]*tau_l
		alphas_w=list(np.array(alphas_w, dtype=np.float32))
		alphas_l=list(np.array(alphas_l, dtype=np.float32))
		
		hp_dict = create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, dataset=dataset)
		train_net_lat_cif(Network_Lateral_Cif(name, hp_dict))
	if L==3 and rule=='smep':
		if (guess==0) or (guess==1) or (guess==6):
			alphas_w = np.array([0.128, 0.032, 0.008, 0.002], dtype=np.float32)

		if (guess==2) or (guess==4) or (guess==9): #0.5, 0.375...
			alphas_w=np.zeros(L+1)
			tau_w=0.75
			alphas_w[0]=0.5
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w	
		if (guess==3) or (guess==5): #0.5, 0.25...
			alphas_w=np.zeros(L+1)
			tau_w=0.5
			alphas_w[0]=0.5
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w
		if (guess==7) or (guess==8): #0.25, 0.125
			alphas_w=np.zeros(L+1)
			tau_w=0.25
			alphas_w[0]=0.5
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w
		if (guess==10):
			alphas_w=np.zeros(L+1)
			tau_w=0.75
			alphas_w[0]=0.4
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w
		
		if (guess==11) or (guess==12):
			alphas_w=np.zeros(L+1)
			tau_w=0.75
			alphas_w[0]=0.6
			for l in range(1, 4):
				alphas_w[l]=alphas_w[l-1]*tau_w
			
		if (guess==0) or (guess==4) or (guess==5) or (guess==7) or (guess==10) or (guess==11):			
			tau_l=np.float32(1.5)
			alphas_l=alphas_w[:-1]*tau_l
		if (guess==1) or (guess==2) or (guess==3) or (guess==8) or (guess==9):	
			alphas_l=np.array([0.05, 0.01, 0.005], dtype=np.float32)
		if (guess==6):			
			tau_l=np.float32(3.0)
			alphas_l=alphas_w[:-1]*tau_l	
		if (guess==9):			
			tau_l=np.float32(0.3)
			alphas_l=alphas_w[:-1]*tau_l
		if (guess==12):
			tau_l=np.float32(1.2)
			alphas_l=alphas_w[:-1]*tau_l
		alphas_w=list(np.array(alphas_w, dtype=np.float32))
		alphas_l=list(np.array(alphas_l, dtype=np.float32))
		
		if exp_type=="normal":
			hp_dict = create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, dataset=dataset, beta=np.float32(1.0))
			train_net_lat_smep(Network_Lateral_SMEP(name, hp_dict))
		if exp_type=="clipdiff":
			print "Modified non linearity"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, dataset=dataset, variant="clipdiff")
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
		if exp_type=="alphadiff":
			print "Changing alphas to small values"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=list(np.array([0.05, 0.01, 0.005] , dtype=np.float32, variant="alphadiff")))
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
		if exp_type=="alphatd1":
			print "Alpha Time Dep continual decrease"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, alpha_tdep_type='alphatd1')
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
		if exp_type=="alphatd2":
			print "Alpha Time Dep continual decrease alphaLt = alphaL0/(1+0.5epoch_num)"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, alpha_tdep_type='alphatd2')
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
		
		if exp_type=="cont_alphasmall":
			print "Post 70 epochs, reads in and uses small alphalat (without clipdiff)"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, alpha_tdep_type='cont_alphasmall')
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
		if exp_type=="alpha_segmented":
			print "1-40: SMEP_4, 40-60: SMEP_4/10, 60 onw: SMEP_4/100"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, alpha_tdep_type=exp_type)
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
		
		if exp_type=="alpha_segmented_old_40":
			print "1-40: SMEP_4, 40-60: 0.05, 0.01, 0.005, 60+: 0.005, 0.001, 0.0005"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, alpha_tdep_type=exp_type)
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
		if exp_type=="alpha_segmented_repr":
			print "1-70: SMEP_4, 70+: 0.05, 0.01, 0.005"
			hp_dict=create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, alpha_tdep_type=exp_type)
			train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))

	return


def run_experiment(exp_type='normal'):
	if exp_type=='normal':
		if sys.argv[3]=='smep':
			rule='smep'
			subdir='SMEP'
		else:
			rule='eplat'
			subdir='EPonly_Lat'
		
		if (sys.argv[2]=='new'):
			dirname ='Cluster_runs/'+subdir+'/Net3/'
		elif (sys.argv[2]=='continuing'):
			dirname='Continuing_runs/'
		else:
			print ('Error, incorrect second arg')
		run_multiple(dirname, L=3, rule=rule, guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	elif exp_type=='clipdiff': #this and all subsequent exp_types are for SMEP 
		dirname='Mod_Experiments/clipdiff/'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	elif exp_type=='alphadiff': #this and all subsequent exp_types are for SMEP 
		dirname='Mod_Experiments/alphadiff'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	elif (exp_type=='alphatd1'):
		dirname='Mod_Experiments/alphatd1_wprint/'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	elif (exp_type=='alphatd2'):
		dirname='Mod_Experiments/alphatd2_wprint/'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)

	elif (exp_type=='cont_alphasmall'):
		dirname='Mod_Experiments/cont_alphasmall/'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	elif(exp_type=='alpha_segmented'):
		dirname='Fast_Net3_reruns/'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	elif (exp_type=='alpha_segmented_old_40'):
		dirname='Fast_Net3_reruns/alpha40postsmall/'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	elif (exp_type=='alpha_segmented_repr'):
		dirname='Fast_Net3_reruns/alpha70postsmall/'
		run_multiple(dirname, L=3, rule='smep', guess=int(sys.argv[1]), dataset=sys.argv[4], exp_type=exp_type)
	else:
		print 'Error: Unimplemented experiment type ', exp_type
		
		

if __name__=='__main__':
	
	#PART1: Code for 1 HL, single run, specifying hyperparams
	dirname='Results/' #CHANGE DIRECTORY CHOICE HERE, .save file will be stored in this directory
	name = dirname + 'smep'
	hpd1 = create_hyp_param_combination(hidden_sizes = [500],
		n_epochs=40,
		batch_size=20,
		n_it_neg=20,
		n_it_pos=4,
		epsilon=np.float32(.5),
		beta=np.float32(100.),
		beta_reg_bool=False,
		alpha_tdep_type='constant',
		dataset="mnist",
		variant="normal",
		alphas_fwd= [np.float32(0.2), np.float32(0.1)], #0.5, 0.375
        alphas_unsuper=[np.float32(0.001),np.float32(0.0005)],  #Update for unsupervised
		alphas_lat=[np.float32(0.01)], #CHANGE ALPHA LATERAL HERR, 0.01
        super_thr = np.float32(0.5),  # The ratio of superivsed samples, in the range of [0,1]
	)
	train_net_smep_mod(Network_SMEP_Mod(name, hpd1))
	
	
	#PART2: Code for 3HL, different variants, predecided learning rate combinations
	#run_experiment('alpha_segmented_repr')
	#run_experiment('alpha_segmented_old_40')
	#run_experiment('alpha_segmented')
	#run_experiment('cont_alphasmall')
	#run_experiment('alphatd2')
	#run_experiment('alphatd1')
	#run_experiment('alphadiff')