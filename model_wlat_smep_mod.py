import cPickle
import numpy as np
import os
import theano
import theano.tensor as T
from itertools import chain, izip_longest
from theano.ifelse import ifelse

from external_world import External_World



def rho(s):
    return T.clip(s, 0., 1.)
    #return T.nnet.sigmoid(4.*s-2.)
def rho_diff(s):
    return T.clip(s, -1., 1.)


class Network(object):

    def __init__(self, name, hyperparameters=dict()):

        self.path = name + ".save"

        # LOAD/INITIALIZE PARAMETERS
        self.biases, self.weights, self.hyperparameters, self.training_curves = self.__load_params(hyperparameters)

        # LOAD EXTERNAL WORLD (=DATA)
        self.external_world = External_World()

        # INITIALIZE PERSISTENT PARTICLES
        dataset_size = self.external_world.size_dataset
        layer_sizes = [28 * 28] + self.hyperparameters["hidden_sizes"] + [10]
        values = [np.zeros((dataset_size, layer_size), dtype=theano.config.floatX) for layer_size in layer_sizes[1:]] #values of the nodes, 1hl to output
        self.persistent_particles = [theano.shared(value, borrow=True) for value in values] #1HL-OP

        # LAYERS = MINI-BACTHES OF DATA + MINI-BACTHES OF PERSISTENT PARTICLES
        batch_size = self.hyperparameters["batch_size"]
        self.index = theano.shared(np.int32(0), name='index')  # index of a mini-batch
        self.super_flag = False   # default superivsed flag for the samples

        self.x_data = self.external_world.x[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data = self.external_world.y[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10)

        self.layers = [self.x_data] + [particle[self.index * batch_size: (self.index + 1) * batch_size] for particle in
                                       self.persistent_particles] #R?

        # BUILD THEANO FUNCTIONS
        self.rho=self.__build_nonlinfunc()
        self.change_mini_batch_index = self.__build_change_mini_batch_index()
        self.measure = self.__build_measure()
        self.measure_verbose = self.__build_measure_verbose()
        self.free_phase_verbose = self.__build_free_phase_verbose()
        self.free_phase = self.__build_free_phase()
        self.weakly_clamped_phase = self.__build_weakly_clamped_phase()
        self.unsupervised_phase = self.__build_unsupervised_phase()



    def save_params(self):
        f = file(self.path, 'wb')
        biases_fwd_values = [b.get_value() for b in self.biases['fwd']]
        weights_fwd_values, weights_lat_values = [W.get_value() for W in self.weights['fwd']], [W.get_value() for W in self.weights['lat']]
        to_dump = biases_fwd_values, weights_fwd_values, weights_lat_values, self.hyperparameters, self.training_curves
        cPickle.dump(to_dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
    def __build_nonlinfunc(self):
        var=self.hyperparameters["variant"]
        if var=='clipdiff':
            print "Clipdiff initialization"
            return rho_diff
        else:
            return rho
        
            

    def __load_params(self, hyperparameters): #R

        hyper = hyperparameters

        # Glorot/Bengio weight initialization
        def initialize_layer(n_in, n_out, pd_bool=False):
            if pd_bool==False:
                rng = np.random.RandomState()
                W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )

            else: #n_in = n_out since lateral
                rng=np.random.RandomState()
                W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                W_values = np.matmul(W_values, W_values.transpose())
            return W_values

        if os.path.isfile(self.path): #loads weights from the .save file for all subsequent epochs
            f = file(self.path, 'rb')
            biases_fwd_values, weights_fwd_values, weights_lat_values, hyperparameters, training_curves = cPickle.load(f)
            f.close()
            for k, v in hyper.iteritems(): #replaces the loaded hyperparams from the .save file with the hyperparameters arg
                hyperparameters[k] = v
            print ('Reading in weights from stored network file')
        else: #initializes, R
            layer_sizes = [28 * 28] + hyperparameters["hidden_sizes"] + [10] #R
            biases_fwd_values = [np.zeros((size,), dtype=theano.config.floatX) for size in layer_sizes[1:]] #R
            weights_fwd_values = [initialize_layer(size_pre, size_post) for size_pre, size_post in
                              zip(layer_sizes[:-1], layer_sizes[1:])] #R
            #biases_lat_values = [np.zeros((size,), dtype=theano.config.floatX) for size in layer_sizes[1:(-1)]]
            weights_lat_values = [initialize_layer(size, size, pd_bool=True) for size in layer_sizes[1:(-1)]] #recurrent in all but input and output
            print ('W norm: ', np.linalg.norm(weights_fwd_values[0]), np.linalg.norm(weights_fwd_values[1]), ' W norm perelem: ', np.linalg.norm(weights_fwd_values[0])/(784*500), np.linalg.norm(weights_fwd_values[1])/(500*10))
            print ('M norm: ', np.linalg.norm(weights_lat_values[0]), np.linalg.norm(weights_lat_values[0])/(500*500))
            training_curves = dict()
            training_curves["training error"] = list()
            training_curves["validation error"] = list()
        weights_all = dict({'fwd': [theano.shared(value=value, borrow=True) for value in weights_fwd_values],
                            'lat': [theano.shared(value=value, borrow=True) for value in weights_lat_values]})
        biases_all = dict({'fwd': [theano.shared(value=value, borrow=True) for value in biases_fwd_values]})

        return biases_all, weights_all, hyperparameters, training_curves
    
    def change_batch_super_flag(self, new_flag):
        self.super_flag = new_flag
    # SET INDEX OF THE MINI BATCH
    def __build_change_mini_batch_index(self):

        index_new = T.iscalar("index_new")
        #flag_new = T.bscalar("super_flag")

        change_mini_batch_index = theano.function(
            inputs=[index_new],
            outputs=[],
            updates=[(self.index, index_new)]
        )

        return change_mini_batch_index

    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self, layers, longbool=0): #Eqn 1
        squared_norm = sum([T.batched_dot(self.rho(layer), self.rho(layer)) for layer in layers]) / 2.
        linear_terms = - sum([T.dot(self.rho(layer), b) for layer, b in zip(layers[1:], self.biases['fwd'])]) #R
        #linear_terms -= sum([T.dot(rho(layer), b) for layer, b in zip(layers[1:(-1)], self.biases['lat'])])
        quadratic_terms = - sum([T.batched_dot(T.dot(self.rho(pre), W), self.rho(post)) for pre, W, post in
                                 zip(layers[:-1], self.weights['fwd'], layers[1:])]) #R
        # quad_lat = sum([T.batched_dot(T.dot(self.rho(node), W), self.rho(node)) for W, node in
        #                          zip(self.weights['lat'], layers[1:(-1)])]) / 2.0
        quad_lat = sum([T.batched_dot(T.dot(self.rho(node), W), self.rho(node)) for W, node in
                        zip(self.weights['lat'], layers[1:])]) / 2.0
            #CHECK IF ONE SHOULD BE THE TRANSPOSE, AND IF CONSTANT FACTOR OF 0.5
        if longbool==0:
            return squared_norm + linear_terms + quadratic_terms +quad_lat
        else:
            return (squared_norm + linear_terms + quadratic_terms + quad_lat), squared_norm, linear_terms, quadratic_terms, quad_lat

    # COST FUNCTION, DENOTED BY C
    def __cost(self, layers):
        return ((layers[-1] - self.y_data_one_hot) ** 2).sum(axis=1)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_energy(self, layers, beta):
        return self.__energy(layers) + beta * self.__cost(layers)

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __build_measure(self):

        E = T.mean(self.__energy(self.layers))
        C = T.mean(self.__cost(self.layers))
        y_prediction = T.argmax(self.layers[-1], axis=1)
        error = T.mean(T.neq(y_prediction, self.y_data))

        measure = theano.function(
            inputs=[],
            outputs=[E, C, error]
        )

        return measure

    def __build_measure_verbose(self):
        e, sq_n, lin_term, quad_term, quad_lat = self.__energy(self.layers, 1)
        E = T.mean(e)
        C = T.mean(self.__cost(self.layers))
        y_prediction = T.argmax(self.layers[-1], axis=1)
        error = T.mean(T.neq(y_prediction, self.y_data))
        Sq_n = T.mean(sq_n)
        Lin_term = T.mean(lin_term)
        Quad_term = T.mean(quad_term)
        Quad_lat = T.mean(quad_lat)

        measure = theano.function(inputs=[], outputs=[E, C, error, Sq_n, Lin_term, Quad_term, Quad_lat], on_unused_input='warn')

        return measure

    def __build_free_phase(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon = T.fscalar('epsilon')

        def step(*layers): #Eq8, beta=0
            E_sum = T.sum(self.__energy(layers))
            layers_dot = T.grad(-E_sum, list(layers))  # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]] + [self.rho(layer + epsilon * dot) for layer, dot in
                                        zip(layers, layers_dot)][1:]
            return layers_new
        #Neural Dynamics
        (layers, updates) = theano.scan(
            step,
            outputs_info=self.layers, #initial output
            n_steps=n_iterations #number of iterative updates of si
        )
        layers_end = [layer[-1] for layer in layers] #si after updating

        for particles, layer, layer_end in zip(self.persistent_particles, self.layers[1:], layers_end[1:]):
            updates[particles] = T.set_subtensor(layer, layer_end) #replaces old subtensors of layer with layer_end, assigns to updates[particles]

        free_phase = theano.function( #SV Inputs: particles (node values), the particles are updated according to updates calculated from theano scan
            inputs=[n_iterations, epsilon],
            outputs=[],
            updates=updates
        )

        return free_phase

    def __build_free_phase_verbose(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon = T.fscalar('epsilon')

        def step(*layers): #Eq8, beta=0
            E_sum = T.sum(self.__energy(layers))
            layers_dot = T.grad(-E_sum, list(layers))  # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]] + [self.rho(layer + epsilon * dot) for layer, dot in
                                        zip(layers, layers_dot)][1:]
            return layers_new
        #Neural Dynamics
        (layers, updates) = theano.scan(
            step,
            outputs_info=self.layers, #initial output
            n_steps=n_iterations #number of iterative updates of si
        )
        layers_end = [layer[-1] for layer in layers] #si after updating

        for particles, layer, layer_end in zip(self.persistent_particles, self.layers[1:], layers_end[1:]):
            updates[particles] = T.set_subtensor(layer, layer_end) #replaces old subtensors of layer with layer_end, assigns to updates[particles]

        free_phase = theano.function( #SV Inputs: particles (node values), the particles are updated according to updates calculated from theano scan
            inputs=[n_iterations, epsilon],
            outputs=[],
            updates=updates
        )
        return free_phase
    
    def __build_weakly_clamped_phase(self):
        eps_norm=np.float32(np.power(10.0, -6.))
        n_iterations = T.iscalar('n_iterations')
        epsilon = T.fscalar('epsilon')
        beta = T.fscalar('beta')
        alphas = [T.fscalar("alpha_W" + str(r + 1)) for r in range(len(self.weights['fwd']))] + [
            T.fscalar("alpha_L" + str(r + 1)) for r in range(len(self.weights['lat']))]

        alphas_fwd = alphas[:len(self.weights['fwd'])]
        alphas_lat = alphas[len(self.weights['fwd']):]

        # Neural Dynamics
        def step(*layers): #Eq42
            F_sum = T.sum(self.__total_energy(layers, beta))
            layers_dot = T.grad(-F_sum, list(layers))  # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]] + [self.rho(layer + epsilon * dot) for layer, dot in
                                        zip(layers, layers_dot)][1:]
            return layers_new

        (layers, updates) = theano.scan(
            step,
            outputs_info=self.layers,
            n_steps=n_iterations
        )
        layers_weakly_clamped = [layer[-1] for layer in layers]

        #R below
        e_free, squared_norm, linear_terms, quadratic_terms, quad_lat = self.__energy(self.layers, 1) #These are the layers post Free relaxation
        E_mean_free = T.mean(e_free)
        e_wc, squared_norm_wc, linear_terms_wc, quadratic_terms_wc, quad_lat_wc  = self.__energy(layers_weakly_clamped, 1)
        E_mean_weakly_clamped = T.mean(e_wc)
        biases_fwd_dot = T.grad((E_mean_weakly_clamped - E_mean_free) / beta, self.biases['fwd'],
                            consider_constant=layers_weakly_clamped) #Eq10
        #biases_lat_dot = T.grad((E_mean_weakly_clamped - E_mean_free) / beta, self.biases['lat'],
        #                    consider_constant=layers_weakly_clamped) #Eq10
        weights_fwd_dot = T.grad((E_mean_weakly_clamped - E_mean_free) / beta, self.weights['fwd'],
                             consider_constant=layers_weakly_clamped)
        weights_lat_dot = T.grad(E_mean_weakly_clamped / beta, self.weights['lat'],
                                 consider_constant=layers_weakly_clamped) #SMEP


        #biases_lat_new = [b - alpha * dot for b, alpha, dot in zip(self.biases['lat'], alphas_lat, biases_lat_dot)]
        biases_fwd_new = [b - alpha * dot for b, alpha, dot in zip(self.biases['fwd'], alphas_fwd, biases_fwd_dot)]
        weights_fwd_new = [W - alpha * dot for W, alpha, dot in zip(self.weights['fwd'], alphas_fwd, weights_fwd_dot)] #Eq13
        weights_lat_new = [W + alpha * (dot - W/(2.0*np.abs(beta))) for W, alpha, dot in
                           zip(self.weights['lat'], alphas_lat, weights_lat_dot)]  #SMEP
        Delta_log = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['fwd'],weights_fwd_new)]+ [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['lat'],weights_lat_new)]


        #Delta_log = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['fwd'],weights_fwd_new)]+ [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['lat'],weights_lat_new)]+[T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean()+eps_norm ) for W,W_new in zip(self.weights['lat'],weights_lat_new)]
        Signed_delta_log = [((W_new - W)/np.abs(W)).mean() for W, W_new in zip(self.weights['fwd'], weights_fwd_new)]+[((W_new - W)/np.abs(W)).mean() for W, W_new in zip(self.weights['lat'], weights_lat_new)]

        #ENSURE UPDATES IS IN A FORM WHERE CAN BE INDEXED IN BELOW FASHION
        #CHECK BELOW

        for bias, bias_new in zip(self.biases['fwd'], biases_fwd_new):
            updates[bias] = bias_new

        for weight, weight_new in zip(self.weights['fwd'], weights_fwd_new):
            updates[weight] = weight_new

        for weight, weight_new in zip(self.weights['lat'], weights_lat_new):
            updates[weight] = weight_new

        #out= T.as_tensor_variable([Delta_log])
        #out_b = T.as_tensor_variable([biases_fwd_new])
        #out_wf = T.as_tensor_variable([weights_fwd_new])
        #out_wl = T.as_tensor_variable([weights_lat_new])

        weakly_clamped_phase = theano.function( #SV Inputs: network weights, the updates are applied to the weights
            inputs=[n_iterations, epsilon, beta]+alphas,
            outputs=Delta_log,
            #outputs=[Delta_log[0], Delta_log[1], Delta_log[2], Signed_delta_log[0], Signed_delta_log[1], Signed_delta_log[2], biases_fwd_dot[0], biases_fwd_dot[1], weights_fwd_dot[0], weights_fwd_dot[1], weights_lat_dot[0], E_mean_free, E_mean_weakly_clamped,
            #         squared_norm, squared_norm_wc, linear_terms, linear_terms_wc, quadratic_terms, quadratic_terms_wc, quad_lat, quad_lat_wc],
            updates=updates
        )

        return weakly_clamped_phase

    def __build_unsupervised_phase(self):
        """
        This function update the weights and biases at the unsuperivsed phase
        """
        n_iterations = T.iscalar('n_iterations')
        epsilon = T.fscalar('epsilon')
        beta = T.fscalar('beta')

        def step(*layers): #Eq8, beta=0
            E_sum = T.sum(self.__energy(layers))
            layers_dot = T.grad(-E_sum, list(layers))  # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]] + [self.rho(layer + epsilon * dot) for layer, dot in
                                        zip(layers, layers_dot)][1:]
            return layers_new
        #Neural Dynamics
        (layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers, #initial output
            n_steps=n_iterations #number of iterative updates of si
        )
        layers_end = [layer[-1] for layer in layers] #si after updating
        #layers_unsuper = [layer[-1] for layer in layers]

        #for unlabeled data, don't update persistent particles
        for particles, layer, layer_end in zip(self.persistent_particles, self.layers[1:], layers_end[1:]):
            updates[particles] = T.set_subtensor(layer, layer_end)         
        
        alphas = [T.fscalar("alpha_W" + str(r + 1)) for r in range(len(self.weights['fwd']))] + [
            T.fscalar("alpha_L" + str(r + 1)) for r in range(len(self.weights['lat']))]
        alphas_fwd = alphas[:len(self.weights['fwd'])]
        alphas_lat = alphas[len(self.weights['fwd']):]

        #R below
        e_free, squared_norm, linear_terms, quadratic_terms, quad_lat = self.__energy(layers_end, 1) #These are the layers post Free relaxation
        E_mean_free = T.mean(e_free)
        
        # Update the weights and biases based on the supervised flag
        biases_fwd_dot = T.grad(E_mean_free / beta, self.biases['fwd'], consider_constant=layers_end) #Eq10
        weights_fwd_dot = T.grad(E_mean_free / beta, self.weights['fwd'], consider_constant=layers_end)
        weights_lat_dot = T.grad(E_mean_free / beta, self.weights['lat'], consider_constant=layers_end) #SMEP

        biases_fwd_new = [b - alpha * (dot + b/np.abs(beta)) for b, alpha, dot in zip(self.biases['fwd'], alphas_fwd, biases_fwd_dot)]
        weights_fwd_new = [W - alpha * (dot + W/np.abs(beta)) for W, alpha, dot in zip(self.weights['fwd'], alphas_fwd, weights_fwd_dot)] #Eq13
        weights_lat_new = [W + alpha * (dot - W/(2.0*np.abs(beta))) for W, alpha, dot in
                           zip(self.weights['lat'], alphas_lat, weights_lat_dot)]  #SMEP
        Delta_log = [T.sqrt(((W_new - W) ** 2).mean()) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['fwd'],weights_fwd_new)] + [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['lat'],weights_lat_new)]
        
        #updates = dict()
        for bias, bias_new in zip(self.biases['fwd'], biases_fwd_new):
            updates[bias] = bias_new

        for weight, weight_new in zip(self.weights['fwd'], weights_fwd_new):
            updates[weight] = weight_new

        for weight, weight_new in zip(self.weights['lat'], weights_lat_new):
            updates[weight] = weight_new

        unsupervised_phase = theano.function( #SV Inputs: network weights, the updates are applied to the weights
            inputs=[n_iterations, epsilon, beta] + alphas,
            outputs=Delta_log,
            #on_unused_input='ignore',
            updates=updates
        )

        return unsupervised_phase