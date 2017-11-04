from load import mnist
import numpy as np

from matplotlib import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# 1 encoder, decoder and a softmax layer

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

trX, teX, trY, teY = mnist()

# We use first 12000 from MNIST train data as train dataset, and first 2000 from test data
trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

x = T.fmatrix('x')  
d = T.fmatrix('d')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# Training Parameters to be used
corruption_level=0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128

# 1. train 1st hidden layer (E1, D1) for encoder decoder
# X -> E1 -> D1
# 2. train 2nd hidden (E2, D2) layer
# X -> E1 -> E2 -> D2 -> D1
# 3. train 3rd hidden (E3, D3) layer
# X -> E1 -> E2 -> E3 -> D3 -> D2 -> D1
# Train 5 layer NN with softmax as output layer
# X -> E1 -> E2 -> E3 -> O
while True:
    qn = str(input("Pick a qn from from 1 to 3:  \nNote: you need to run qn1 to train hidden layers before running qn2\n"))
    if qn.lower() not in ('1','2','3'):
        print("Not an appropriate choice.")
    #===========================================
    #  Qn 1
    # Design stacked denoising autoencoder with three hidden layers
    # - plot learning curves, reconstruction errors on training data for each layer
    # - plot 100 samples of weights learnt at each layer as images
    # - for 100 representative images, plot reconstructed images, hidden layer activation
    #===========================================
    elif qn.lower() == '1':

        # For Hidden layer 1 (E1, D1)
        W_h1 = init_weights(28*28, 900)
        b_h1 = init_bias(900)
        b_h1_prime = init_bias(28*28)
        W_h1_prime = W_h1.transpose()

        # For Hidden layer 2 (E2, D2)
        W_h2 = init_weights(900, 625)
        b_h2 = init_bias(625)
        b_h2_prime = init_bias(900)
        W_h2_prime = W_h2.transpose() 
        # For Hidden layer 3 (E3, D3)
        W_h3 = init_weights(625, 400)
        b_h3 = init_bias(400)
        b_h3_prime = init_bias(625)
        W_h3_prime = W_h3.transpose() 
        # For softmax layer
        W_s = init_weights(400, 10)
        b_s = init_bias(10)

        # Corrupt the input data
        tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                                      dtype=theano.config.floatX)*x

        # For training hidden layer 1
        y1 = T.nnet.sigmoid(T.dot(x, W_h1) + b_h1)
        z1 = T.nnet.sigmoid(T.dot(y1, W_h1_prime) + b_h1_prime)
        cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))

        params1 = [W_h1, b_h1, b_h1_prime]
        grads1 = T.grad(cost1, params1)
        updates1 = [(param1, param1 - learning_rate * grad1)
                   for param1, grad1 in zip(params1, grads1)]
        train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)
        encoder_1 = theano.function(inputs=[x], outputs = y1, allow_input_downcast=True)
        decoder_1 = theano.function(inputs=[y1], outputs = z1, allow_input_downcast=True)


        # For training hidden layer 2
        y2 = T.nnet.sigmoid(T.dot(y1, W_h2) + b_h2)
        z2 = T.nnet.sigmoid(T.dot(y2, W_h2_prime) + b_h2_prime)
        cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))

        params2 = [W_h2, b_h2, b_h2_prime]
        grads2 = T.grad(cost2, params2)
        updates2 = [(param, param - learning_rate * grad)
                   for param, grad in zip(params2, grads2)]
        train_da2 = theano.function(inputs=[y1], outputs = cost2, updates = updates2, allow_input_downcast = True)
        encoder_2 = theano.function(inputs=[y1], outputs = y2, allow_input_downcast=True)
        decoder_2 = theano.function(inputs=[y2], outputs = z2, allow_input_downcast=True)


        # For training hidden layer 3
        y3 = T.nnet.sigmoid(T.dot(y2, W_h3) + b_h3)
        z3 = T.nnet.sigmoid(T.dot(y3, W_h3_prime) + b_h3_prime)
        cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))

        params3 = [W_h3, b_h3, b_h3_prime]
        grads3 = T.grad(cost3, params3)
        updates3 = [(param, param - learning_rate * grad)
                   for param, grad in zip(params3, grads3)]
        train_da3 = theano.function(inputs=[y2], outputs = cost3, updates = updates3, allow_input_downcast = True)
        encoder_3 = theano.function(inputs=[y2], outputs = y3, allow_input_downcast=True)
        decoder_3 = theano.function(inputs=[y3], outputs = z3, allow_input_downcast=True)

        # For training Five-layer feed forward neural network
        p_y_s = T.nnet.softmax(T.dot(y3, W_s)+b_s)
        y_s = T.argmax(p_y_s, axis=1)
        cost_s = T.mean(T.nnet.categorical_crossentropy(p_y_s, d))
        # Update all relevant weights and bias in 5 layer feed forward network
        params_s = [W_h1, b_h1, W_h2, b_h2, W_h3, b_h3, W_s, b_s]
        grads_s = T.grad(cost_s, params_s)
        updates_s = [(param, param - learning_rate * grad)
                   for param, grad in zip(params_s, grads_s)]
        train_ffn = theano.function(inputs=[x, d], outputs = cost_s, updates = updates_s, allow_input_downcast = True)
        test_ffn = theano.function(inputs=[x], outputs = y_s, allow_input_downcast=True)

        # Train 1st hidden layer
        print('training dae1 ...')
        d1 = []
        for epoch in range(training_epochs):
            # go through trainng set
            c1 = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c1.append(train_da1(trX[start:end]))
            d1.append(np.mean(c1, dtype='float64'))
            print(d1[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d1)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('hiddenlayer1_cross-entropy.png')

        # get trained w and b
        tr_W_h1 = W_h1.get_value()
        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(tr_W_h1[:,i].reshape(28,28))
        pylab.savefig('hiddenlayer1_weights.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(trX[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('input.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(decoder_1(encoder_1(trX))[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('output_da1.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(encoder_1(trX)[i,:].reshape(30,30))
        #pylab.title('hidden layer activations 1')
        pylab.savefig('hidden_activations_1.png')

        # Train 2nt hidden layer
        print('training dae2 ...')
        d2 = []
        for epoch in range(training_epochs):
            # go through trainng set
            c2 = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c2.append(train_da2(encoder_1(trX[start:end])))
            d2.append(np.mean(c2, dtype='float64'))
            print(d2[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d2)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('hiddenlayer2_cross-entropy.png')

        # get trained w and b
        tr_W_h2 = W_h2.get_value()
        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(tr_W_h2[:,i].reshape(30,30))
        pylab.savefig('hiddenlayer2_weights.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(decoder_1(decoder_2(encoder_2(encoder_1(trX))))[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('output_da2.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(encoder_2(encoder_1(trX))[i,:].reshape(25,25))
        #pylab.title('hidden layer activations 2')
        pylab.savefig('hidden_activation_2.png')

        # Train 3rd hidden layer
        print('training dae3 ...')
        d3 = []
        for epoch in range(training_epochs):
            # go through trainng set
            c3 = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c3.append(train_da3(encoder_2(encoder_1(trX[start:end]))))
            d3.append(np.mean(c3, dtype='float64'))
            print(d3[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d3)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('hiddenlayer3_cross-entropy.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(decoder_1(decoder_2(decoder_3(encoder_3(encoder_2(encoder_1(trX))))))[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('output_da3.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(encoder_3(encoder_2(encoder_1(trX)))[i,:].reshape(20,20))
        #pylab.title('hidden layer activations 3')
        pylab.savefig('hidden_activation_3.png')

        # get trained w and b
        tr_W_h3 = W_h3.get_value()
        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(tr_W_h3[:,i].reshape(25,25))
        pylab.savefig('hiddenlayer3_weights.png')
    #===========================================
    #  Qn 2
    # Train a 5-layer FFN using the three layers learnt in qn1
    # - plot training errors and test accuracy during training
    #===========================================
    elif qn.lower() == '2':
        print('\ntraining ffn ...')
        d, a = [], []
        for epoch in range(training_epochs):
            # go through trainng set
            c = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c.append(train_ffn(trX[start:end], trY[start:end]))
            d.append(np.mean(c, dtype='float64'))
            a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
            print(a[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('ffn_training_errors.png')

        pylab.figure()
        pylab.plot(range(training_epochs), a)
        pylab.xlabel('iterations')
        pylab.ylabel('test accuracy')
        pylab.savefig('ffn_test_accuracy.png')

        w_s = W_s.get_value()
        pylab.figure()
        pylab.gray()
        pylab.axis('off'); pylab.imshow(w_s)
        pylab.savefig('softmax_weights.png')
    #===========================================
    #  Qn 3
    # Repeat step 1 and 2 introducing momentum term for GD and sparcity constraint to cost function
    # Momentum parameter 0.1, penalty param 0.5 and sparsity param 0.05
    # Compare results with that from part 1 and 2
    #===========================================    
    elif qn.lower() == '3':
        print("\n repeating qn 1 and 2 with momentum term and sparsity constraint")
        
        # Reinitialise weights and bias for qn 3
        # For Hidden layer 1 (E1, D1)
        W_h1 = init_weights(28*28, 900)
        b_h1 = init_bias(900)
        b_h1_prime = init_bias(28*28)
        W_h1_prime = W_h1.transpose()

        # For Hidden layer 2 (E2, D2)
        W_h2 = init_weights(900, 625)
        b_h2 = init_bias(625)
        b_h2_prime = init_bias(900)
        W_h2_prime = W_h2.transpose() 
        # For Hidden layer 3 (E3, D3)
        W_h3 = init_weights(625, 400)
        b_h3 = init_bias(400)
        b_h3_prime = init_bias(625)
        W_h3_prime = W_h3.transpose() 
        # For softmax layer
        W_s = init_weights(400, 10)
        b_s = init_bias(10)

        momentum = 0.1
        beta = 0.5
        rho = 0.05 #sparsity parameter

        # function to update cost with SGD
        def sgd_momentum(cost,  params, lr=learning_rate, decay=0.00001, momentum=0.1):              
            grads = T.grad(cost=cost,   wrt=params)             
            updates = []              
            for p, g in zip(params, grads):                         
                v = theano.shared(p.get_value())                                
                v_new = momentum*v - (g + decay*p) * lr
                updates.append([p, p + v_new])                         
                updates.append([v, v_new])                             
            return updates

        # For training hidden layer 1
        y1 = T.nnet.sigmoid(T.dot(x, W_h1) + b_h1)
        z1 = T.nnet.sigmoid(T.dot(y1, W_h1_prime) + b_h1_prime)
        cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1)) +  beta*T.shape(y1)[1]*(rho*T.log(rho)+ (1-rho)*T.log(1-rho)) - beta*rho*T.sum(T.log(T.mean(y1, axis=0)+1e-6)) - beta*(1-rho)*T.sum(T.log(1-T.mean(y1, axis=0)+1e-6))

        params1 = [W_h1, b_h1, b_h1_prime]
        train_da1 = theano.function(inputs=[x], outputs = cost1, updates = sgd_momentum(cost1,params1), allow_input_downcast = True)
        encoder_1 = theano.function(inputs=[x], outputs = y1, allow_input_downcast=True)
        decoder_1 = theano.function(inputs=[y1], outputs = z1, allow_input_downcast=True)

        # For training hidden layer 2
        y2 = T.nnet.sigmoid(T.dot(y1, W_h2) + b_h2)
        z2 = T.nnet.sigmoid(T.dot(y2, W_h2_prime) + b_h2_prime)
        cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))+  beta*T.shape(y2)[1]*(rho*T.log(rho)+ (1-rho)*T.log(1-rho)) - beta*rho*T.sum(T.log(T.mean(y2, axis=0)+1e-6)) - beta*(1-rho)*T.sum(T.log(1-T.mean(y2, axis=0)+1e-6))

        params2 = [W_h2, b_h2, b_h2_prime]
        train_da2 = theano.function(inputs=[y1], outputs = cost2, updates = sgd_momentum(cost2,params2), allow_input_downcast = True)
        encoder_2 = theano.function(inputs=[y1], outputs = y2, allow_input_downcast=True)
        decoder_2 = theano.function(inputs=[y2], outputs = z2, allow_input_downcast=True)


        # For training hidden layer 3
        y3 = T.nnet.sigmoid(T.dot(y2, W_h3) + b_h3)
        z3 = T.nnet.sigmoid(T.dot(y3, W_h3_prime) + b_h3_prime)
        cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))+  beta*T.shape(y3)[1]*(rho*T.log(rho)+ (1-rho)*T.log(1-rho)) - beta*rho*T.sum(T.log(T.mean(y3, axis=0)+1e-6)) - beta*(1-rho)*T.sum(T.log(1-T.mean(y3, axis=0)+1e-6))

        params3 = [W_h3, b_h3, b_h3_prime]
        train_da3 = theano.function(inputs=[y2], outputs = cost3, updates = sgd_momentum(cost3,params3), allow_input_downcast = True)
        encoder_3 = theano.function(inputs=[y2], outputs = y3, allow_input_downcast=True)
        decoder_3 = theano.function(inputs=[y3], outputs = z3, allow_input_downcast=True)

        # For training Five-layer feed forward neural network
        p_y_s = T.nnet.softmax(T.dot(y3, W_s)+b_s)
        y_s = T.argmax(p_y_s, axis=1)
        cost_s = T.mean(T.nnet.categorical_crossentropy(p_y_s, d))
        # Update all relevant weights and bias in 5 layer feed forward network
        params_s = [W_h1, b_h1, W_h2, b_h2, W_h3, b_h3, W_s, b_s]
        grads_s = T.grad(cost_s, params_s)
        updates_s = [(param, param - learning_rate * grad)
                   for param, grad in zip(params_s, grads_s)]
        train_ffn = theano.function(inputs=[x, d], outputs = cost_s, updates = updates_s, allow_input_downcast = True)
        test_ffn = theano.function(inputs=[x], outputs = y_s, allow_input_downcast=True)


        # Train 1st hidden layer
        print('training dae1 ...')
        d1 = []
        for epoch in range(training_epochs):
            # go through trainng set
            c1 = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c1.append(train_da1(trX[start:end]))
            d1.append(np.mean(c1, dtype='float64'))
            print(d1[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d1)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('qn3_hiddenlayer1_cross-entropy.png')

        # get trained w and b
        tr_W_h1 = W_h1.get_value()
        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(tr_W_h1[:,i].reshape(28,28))
        pylab.savefig('qn3_hiddenlayer1_weights.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(trX[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('qn3_input.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(decoder_1(encoder_1(trX))[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('qn3_output_da1.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(encoder_1(trX)[i,:].reshape(30,30))
        #pylab.title('hidden layer activations 1')
        pylab.savefig('qn3_hidden_activations_1.png')

        # Train 2nt hidden layer
        print('training dae2 ...')
        d2 = []
        for epoch in range(training_epochs):
            # go through trainng set
            c2 = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c2.append(train_da2(encoder_1(trX[start:end])))
            d2.append(np.mean(c2, dtype='float64'))
            print(d2[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d2)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('qn3_hiddenlayer2_cross-entropy.png')

        # get trained w and b
        tr_W_h2 = W_h2.get_value()
        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(tr_W_h2[:,i].reshape(30,30))
        pylab.savefig('qn3_hiddenlayer2_weights.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(decoder_1(decoder_2(encoder_2(encoder_1(trX))))[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('qn3_output_da2.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(encoder_2(encoder_1(trX))[i,:].reshape(25,25))
        #pylab.title('hidden layer activations 2')
        pylab.savefig('qn3_hidden_activation_2.png')

        # Train 3rd hidden layer
        print('training dae3 ...')
        d3 = []
        for epoch in range(training_epochs):
            # go through trainng set
            c3 = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c3.append(train_da3(encoder_2(encoder_1(trX[start:end]))))
            d3.append(np.mean(c3, dtype='float64'))
            print(d3[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d3)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('qn3_hiddenlayer3_cross-entropy.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(decoder_1(decoder_2(decoder_3(encoder_3(encoder_2(encoder_1(trX))))))[i,:].reshape(28,28))
        #pylab.title('input image')
        pylab.savefig('qn3_output_da3.png')

        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1);pylab.axis('off'); pylab.imshow(encoder_3(encoder_2(encoder_1(trX)))[i,:].reshape(20,20))
        #pylab.title('hidden layer activations 3')
        pylab.savefig('qn3_hidden_activation_3.png')

        # get trained w and b
        tr_W_h3 = W_h3.get_value()
        pylab.figure()
        pylab.gray()
        for i in range(100):
            pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(tr_W_h3[:,i].reshape(25,25))
        pylab.savefig('qn3_hiddenlayer3_weights.png')

        print('\ntraining ffn ...')
        d, a = [], []
        for epoch in range(training_epochs):
            # go through trainng set
            c = []
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                c.append(train_ffn(trX[start:end], trY[start:end]))
            d.append(np.mean(c, dtype='float64'))
            a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
            print(a[epoch])

        pylab.figure()
        pylab.plot(range(training_epochs), d)
        pylab.xlabel('iterations')
        pylab.ylabel('cross-entropy')
        pylab.savefig('qn3_ffn_training_errors.png')

        pylab.figure()
        pylab.plot(range(training_epochs), a)
        pylab.xlabel('iterations')
        pylab.ylabel('test accuracy')
        pylab.savefig('qn3_ffn_test_accuracy.png')

        w_s = W_s.get_value()
        pylab.figure()
        pylab.gray()
        pylab.axis('off'); pylab.imshow(w_s)
        pylab.savefig('qn_3softmax_weights.png')