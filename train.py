import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

def map_data_with_classes(classes):

  maxi = 0
  i = 0
  while(i!=len(classes)):
    if(maxi<classes[i]) :
      maxi = classes[i]
    i+=1
  
  cols = maxi + 1
  rows = len(classes)
  matrix = np.zeros((rows,cols))

  j = 0
  while(j!=len(classes)):
    matrix[j][classes[j]] = 1
    j+=1
  return matrix

(train_X,train_Y),(test_X,test_Y) = fashion_mnist.load_data()
train_X = train_X/255
test_X = test_X/255

needed_y_train = train_Y
needed_y_test = test_Y
#flatten 2d image vectors to 1d vectors and treat them as training data

trainX, val_X, trainy, valy = train_test_split(train_X, train_Y, test_size=0.1, random_state=40)

trainX = trainX.reshape(len(trainX),len(trainX[0])*len(trainX[1]))
testX = test_X.reshape(len(test_X),len(test_X[0])*len(test_X[1]))
valX = val_X.reshape(len(val_X),len(val_X[0])*len(val_X[1]))

trainy = map_data_with_classes(trainy)
testy = map_data_with_classes(test_Y)
valiy = map_data_with_classes(valy)

input_layer_size = len(trainX[0])
output_layer_size = len(trainy[0])

def initialize_weights_and_biases(layers,number_hidden_layers = 1,init_type='random'):
  weights = []
  biases = []
  if(init_type == 'random'):
    i = 0
    while(i!=number_hidden_layers+1):
      ws = np.random.normal(0,0.5,(layers[i]['output_size'],layers[i]['input_size']))
      weights.append(ws)

      bs = np.random.normal(0,0.5,(layers[i]['output_size'],1))
      biases.append(bs)
      
      i+=1
  elif(init_type == 'xavier'):
    while(i!=number_hidden_layers+1):
      limit = np.sqrt(6/(layers[i]['output_size'] + layers[i]['input_size']))
      ws = np.random.uniform(-limit, limit, size=(layers[i]['output_size'], layers[i]['input_size']))
      weights.append(ws)

      bs = np.random.uniform(-limit, limit, size=(layers[i]['output_size'], 1))
      biases.append(bs)

  return weights,biases

def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))

def tanh(x):
  return np.tanh(x)

def relu(x):
  return np.maximum(0,x)

def softmax(x):
  y = []
  for i in range(len(x)):
    s = 0
    x[i] -= x[i][np.argmax(x[i])]
    for j in range(len(x[i])):
      s+=(np.exp(x[i][j]))
    y.append(np.exp(x[i])/s)
  return np.array(y)

def cross_entropy(y_hat,y):
  ep = 1e-9
  error = -(np.multiply(y,np.log(y_hat+ep))).sum()/len(y_hat)
  return error

def mean_squared_error(y_hat,y):
  error = np.sum(((y-y_hat)**2)/(2*len(y)))
  return error

def activation_functions(x,activation_function = 'sigmoid') :
  if activation_function == 'sigmoid' :
    return sigmoid(x)
  elif activation_function == 'softmax':
    return softmax(x)
  elif activation_function == 'tanh':
    return tanh(x)
  elif activation_function == 'relu':
    return relu(x)
  else:
    return 'error'

def activation_derivative(x, activation_function="sigmoid"):
    if activation_function == "sigmoid":
      return sigmoid(x)*(1.0-sigmoid(x))
    elif activation_function == "tanh":
      return 1.0-(tanh(x)**2)
    elif activation_function == "relu":
      return 1. * (x>0)
    else:
      return 'error'

def train_accuracy(batch_testy,y_predicted,trainy):
  tot = 0
  j = 0
  c = 0
  while(j!=len(batch_testy)):
    k=0
    while(k!=len(batch_testy[j])):
      tot += 1
      l = 0
      while(l!=len(batch_testy[j][k])): 
        if batch_testy[j][k][l] == 1 :
          index_of_one = l
          break
        l+=1
      l = 0
      maxi = 0
      while(l!=len(y_predicted[j][k])): 
        if y_predicted[j][k][l] > maxi :
          maxi = y_predicted[j][k][l]
          pred_class = l
        l+=1
      if(pred_class == index_of_one):
        c+=1
      k+=1
    j+=1
  return (c/len(trainy))

def test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function):
  a,h = forward_propagation(testX,weights,biases,number_hidden_layers, activation_function, output_function)
  y_pred = h[-1]
  y_predicted = []
  i = 0
  while(i!=len(y_pred)):
    y_predicted.append(np.argmax(y_pred[i]))
    i+=1
  i = 0
  c = 0
  while(i!=len(testy)):
    if y_predicted[i] == testy[i]:
      c+=1
    i+=1
  return c/len(testy)

def val_loss(valX,valy,weights,biases,number_hidden_layers, activation_function, output_function):  
  a,h = forward_propagation(valX,weights,biases,number_hidden_layers, activation_function, output_function)
  y_hat = h[-1]
  ep = 1e-9
  error = -(np.multiply(valy,np.log(y_hat+ep))).sum()/len(y_hat)
  return error

def forward_propagation(batchtrainX,weights,biases,number_hidden_layers,activation_function,output_function):
  a = []
  h = []
  N = len(trainX)

  #initializing a1 with the input
  batch_trainX = np.reshape(batchtrainX,(len(batchtrainX),len(batchtrainX[0])))
  a1 = np.dot(weights[0],batch_trainX.T) + biases[0]
  a.append(a1)
  h1 = activation_functions(a1,activation_function)
  h.append(h1)

  #finding a2 to aL-1 using h1 to hL-2 and h2 to hL-1 using a2 to aL-1
  i = 1
  while(i!=number_hidden_layers):
    an = np.dot(weights[i],h[i-1]) + biases[i]
    a.append(an)
    hn = activation_functions(an,activation_function)
    h.append(hn)
    i+=1
  
  #finding aL and hL as output function differs for this
  aL = np.matmul(weights[number_hidden_layers],h[number_hidden_layers-1]) + biases[number_hidden_layers]
  hL = activation_functions(aL.T,output_function)
  hL = hL.T
  a.append(aL)
  h.append(hL)

  i = 0
  while(i!=number_hidden_layers+1):
    a[i] = a[i].T
    h[i] = h[i].T
    i+=1

  return a,h

def backward_propagation(batch_trainy , batch_trainX ,y_hat , a, h, weights, number_hidden_layers ,derivative_function = 'sigmoid'):
  del_a = {}
  del_W = {}
  del_b = {}
  del_h = {}

  batch_trainy = batch_trainy.reshape(len(batch_trainy),len(batch_trainy[0]))
  
  ep =1e-8
  del_a['a'+ str(number_hidden_layers+1)] = -(batch_trainy-y_hat)
  del_h['h'+ str(number_hidden_layers+1)] = -(batch_trainy/(y_hat+ep))

  i = number_hidden_layers + 1
  while(i!=1):
    del_W['W'+ str(i)] = np.dot(del_a['a' + str(i)].T,h[i-2])/len(batch_trainX)

    del_b['b'+ str(i)] = del_a['a'+ str(i)]
    
    del_h['h'+ str(i-1)] = np.dot(weights[i-1].T , del_a['a' + str(i)].T)

    del_a['a'+str(i-1)] = np.multiply(del_h['h'+str(i-1)],activation_derivative(a[i-2].T,derivative_function))
    del_a['a'+str(i-1)] = del_a['a'+str(i-1)].T

    i-=1
  
  del_W['W'+str('1')] = np.dot(del_a['a1'].T,batch_trainX)
  del_b['b'+str('1')] = del_a['a1']
  
  j = 1
  while(j!=len(del_b)+1):
    k = 0
    l = 0
    li = []
    for k in range(len(del_b['b'+str(j)][0])) :
      sum = 0
      for l in range(len(del_b['b'+str(j)])) :
          sum += del_b['b'+str(j)][l][k]
      li.append(sum/len(batch_trainX))
    li = np.array(li)
    del_b['b'+str(j)] = li.reshape(len(li),1)
    j+=1

  return del_W,del_b

def gradient_descent(trainX, trainy, number_hidden_layers = 1, hidden_layer_size = 4, eta = 0.1, initial_weights = 'random', activation_function = 'sigmoid', epochs = 1, output_function = 'softmax', mini_batch_size=4,loss_function = 'cross_entropy'):

#initialize layers of neural networks
  layers = []
  layer1 = {'input_size' : input_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
  layers.append(layer1)
  
  i=0
  while(i!=number_hidden_layers-1):
    hlayer = {'input_size' : hidden_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
    layers.append(hlayer)
    i+=1
  
  layern = {'input_size' : hidden_layer_size, 'output_size' : output_layer_size, 'function' : output_function}
  layers.append(layern)

#initialize weights and biases

  weights,biases =initialize_weights_and_biases(layers,number_hidden_layers,'random')

  number_batches = len(trainX)/mini_batch_size

  mini_batch_trainX = np.array(np.array_split(trainX, number_batches))
  mini_batch_trainy = np.array(np.array_split(trainy, number_batches))

  train_loss_list = []
  val_loss_list = []
  train_acc_list = []
  val_acc_list = []
  h=None
  j = 0
  while(j!=epochs):
    k=0
    tloss = 0
    vloss = 0
    y_predicted = []
    while(k!=number_batches):
      a,h = forward_propagation(mini_batch_trainX[k],weights,biases,number_hidden_layers, activation_function, output_function)
      y_predicted.append(h[-1])

      if loss_function == 'cross_entropy':
        tloss += cross_entropy(h[-1],mini_batch_trainy[k])
      elif loss_function == 'mse':
        tloss += mean_squared_error(h[-1],mini_batch_trainy[k])
      else:
        print('wrong loss function')
      
      del_W,del_b = backward_propagation(mini_batch_trainy[k],mini_batch_trainX[k],h[-1],a,h,weights,number_hidden_layers ,activation_function)

      i = 0
      while(i!=len(weights)):
        weights[i] = weights[i] - (del_W['W'+str(i+1)]*eta)
        biases[i] = biases[i] - (del_b['b'+str(i+1)]*eta) 
        i+=1
      k+=1
    print(j)
    train_acc = train_accuracy(mini_batch_trainy,y_predicted,trainy)
    print("train_acc : ",train_acc*100)
    print("train_loss : ", tloss/number_batches)
    val_acc = test_accuracy(valX,valy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_acc : ",val_acc*100)
    vloss = val_loss(valX,valiy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_loss : ",vloss)

    train_loss_list.append(tloss/number_batches)
    val_loss_list.append(vloss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    j+=1
  
  plot_lists = [train_loss_list,val_loss_list,train_acc_list,val_acc_list]
  return h[-1],weights,biases,plot_lists

def momentum_based_gradient_descent(trainX, trainy, number_hidden_layers = 1, hidden_layer_size = 4, eta = 0.1, initial_weights = 'random', activation_function = 'sigmoid', epochs = 1, output_function = 'softmax', mini_batch_size=4,loss_function = 'cross_entropy'):
  
  

  #initialize layers of neural networks

  layers = []
  layer1 = {'input_size' : input_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
  layers.append(layer1)
  
  i=0
  while(i!=number_hidden_layers-1):
    hlayer = {'input_size' : hidden_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
    layers.append(hlayer)
    i+=1
  
  layern = {'input_size' : hidden_layer_size, 'output_size' : output_layer_size, 'function' : output_function}
  layers.append(layern)

  #initialize weights and biases
  
  number_batches = len(trainX)/mini_batch_size

  weights,biases = initialize_weights_and_biases(layers,number_hidden_layers,'random')

  mini_batch_trainX = np.array(np.array_split(trainX, number_batches))
  mini_batch_trainy = np.array(np.array_split(trainy, number_batches))

  past_weights = []
  past_biases = []

  beta = 0.9

  i = 0
  while(i!=number_hidden_layers+1):
    past_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    past_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    i+=1

  train_loss_list = []
  val_loss_list = []
  train_acc_list = []
  val_acc_list = []
  h=None
  j = 0
  while(j!=epochs):
    k=0
    tloss = 0
    vloss = 0
    y_predicted = []
    while(k!=number_batches):
      a,h = forward_propagation(mini_batch_trainX[k],weights,biases,number_hidden_layers, activation_function, output_function)
      y_predicted.append(h[-1])

      if loss_function == 'cross_entropy':
        tloss += cross_entropy(h[-1],mini_batch_trainy[k])
      elif loss_function == 'mse':
        tloss += mean_squared_error(h[-1],mini_batch_trainy[k])
      else:
        print('wrong loss function')
        
      del_W,del_b = backward_propagation(mini_batch_trainy[k],mini_batch_trainX[k],h[-1],a,h,weights,number_hidden_layers ,activation_function)

      i = 0
      while(i!=number_hidden_layers+1):
        past_weights[i] = (past_weights[i]*beta) + (del_W['W' + str(i+1)] * eta)
        past_biases[i] = (past_biases[i]*beta) + (del_b['b' + str(i+1)] * eta)

        weights[i] = weights[i]-past_weights[i]
        biases[i] = biases[i]-past_biases[i]

        i+=1

      k+=1
    print(j)
    train_acc = train_accuracy(mini_batch_trainy,y_predicted,trainy)
    print("train_acc : ",train_acc*100)
    print("train_loss : ", tloss/number_batches)
    val_acc = test_accuracy(valX,valy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_acc : ",val_acc*100)
    vloss = val_loss(valX,valiy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_loss : ",vloss)

    train_loss_list.append(tloss/number_batches)
    val_loss_list.append(vloss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    j+=1 
  plot_lists = [train_loss_list,val_loss_list,train_acc_list,val_acc_list]
  return h[-1],weights,biases,plot_lists

def nestrov_accelerated_gradient_descent(trainX, trainy, number_hidden_layers = 1, hidden_layer_size = 4, eta = 0.1, initial_weights = 'random', activation_function = 'sigmoid', epochs = 1, output_function = 'softmax', mini_batch_size=4,loss_function = 'cross_entropy'):
  
  #initialize layers of neural networks

  layers = []
  layer1 = {'input_size' : input_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
  layers.append(layer1)
  
  i=0
  while(i!=number_hidden_layers-1):
    hlayer = {'input_size' : hidden_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
    layers.append(hlayer)
    i+=1
  
  layern = {'input_size' : hidden_layer_size, 'output_size' : output_layer_size, 'function' : output_function}
  layers.append(layern)

  #initialize weights and biases

  weights,biases = initialize_weights_and_biases(layers,number_hidden_layers,'random')

  number_batches = len(trainX)/mini_batch_size

  mini_batch_trainX = np.array(np.array_split(trainX, number_batches))
  mini_batch_trainy = np.array(np.array_split(trainy, number_batches))

  past_weights = []
  past_biases = []

  beta = 0.9

  i = 0
  while(i!=number_hidden_layers+1):
    past_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    past_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    i+=1

  train_loss_list = []
  val_loss_list = []
  train_acc_list = []
  val_acc_list = []
  h=None
  j = 0
  while(j!=epochs):
    k=0
    tloss = 0
    vloss = 0
    y_predicted = []
    while(k!=number_batches):
      l=0
      lookahead_weights = []
      lookahead_biases = []
      while(l!=number_hidden_layers+1):
        lookahead_weights.append(weights[l] - (past_weights[l] * beta))
        lookahead_biases.append(biases[l] - (past_biases[l] * beta))
        l+=1
      a,h = forward_propagation(mini_batch_trainX[k],lookahead_weights,lookahead_biases,number_hidden_layers, activation_function, output_function)
      y_predicted.append(h[-1])

      if loss_function == 'cross_entropy':
        tloss += cross_entropy(h[-1],mini_batch_trainy[k])
      elif loss_function == 'mse':
        tloss += mean_squared_error(h[-1],mini_batch_trainy[k])
      else:
        print('wrong loss function')
      
      del_W,del_b = backward_propagation(mini_batch_trainy[k],mini_batch_trainX[k],h[-1],a,h,lookahead_weights,number_hidden_layers ,activation_function)

      i = 0
      while(i!=number_hidden_layers+1):
        past_weights[i] = (past_weights[i]*beta) + (del_W['W' + str(i+1)] * eta)
        past_biases[i] = (past_biases[i]*beta) + (del_b['b' + str(i+1)] * eta)

        weights[i] = weights[i]-past_weights[i]
        biases[i] = biases[i]-past_biases[i]

        i+=1

      k+=1
    print(j)
    train_acc = train_accuracy(mini_batch_trainy,y_predicted,trainy)
    print("train_acc : ",train_acc*100)
    print("train_loss : ", tloss/number_batches)
    val_acc = test_accuracy(valX,valy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_acc : ",val_acc*100)
    vloss = val_loss(valX,valiy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_loss : ",vloss)

    train_loss_list.append(tloss/number_batches)
    val_loss_list.append(vloss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    j+=1 

  plot_lists = [train_loss_list,val_loss_list,train_acc_list,val_acc_list]
  return h[-1],weights,biases,plot_lists

def rmsprop(trainX, trainy, number_hidden_layers = 1, hidden_layer_size = 4, eta = 0.1, initial_weights = 'random', activation_function = 'sigmoid', epochs = 1, output_function = 'softmax', mini_batch_size=4,loss_function = 'cross_entropy'):

  #initialize layers of neural networks

  layers = []
  layer1 = {'input_size' : input_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
  layers.append(layer1)
  
  i=0
  while(i!=number_hidden_layers-1):
    hlayer = {'input_size' : hidden_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
    layers.append(hlayer)
    i+=1
  
  layern = {'input_size' : hidden_layer_size, 'output_size' : output_layer_size, 'function' : output_function}
  layers.append(layern)

  #initialize weights and biases

  weights,biases = initialize_weights_and_biases(layers,number_hidden_layers,'random')

  number_batches = len(trainX)/mini_batch_size

  mini_batch_trainX = np.array(np.array_split(trainX, number_batches))
  mini_batch_trainy = np.array(np.array_split(trainy, number_batches))

  v_weights = []
  v_biases = []

  beta = 0.9
  ep = 1e-3

  i = 0
  while(i!=number_hidden_layers+1):
    v_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    v_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    i+=1

  train_loss_list = []
  val_loss_list = []
  train_acc_list = []
  val_acc_list = []
  h=None
  j = 0
  while(j!=epochs):
    k=0
    tloss = 0
    vloss = 0
    y_predicted = []
    while(k!=number_batches):
      l=0
      a,h = forward_propagation(mini_batch_trainX[k],weights,biases,number_hidden_layers, activation_function, output_function)
      y_predicted.append(h[-1])

      if loss_function == 'cross_entropy':
        tloss += cross_entropy(h[-1],mini_batch_trainy[k])
      elif loss_function == 'mse':
        tloss += mean_squared_error(h[-1],mini_batch_trainy[k])
      else:
        print('wrong loss function')
      
      del_W,del_b = backward_propagation(mini_batch_trainy[k],mini_batch_trainX[k],h[-1],a,h,weights,number_hidden_layers ,activation_function)

      i = 0
      while(i!=number_hidden_layers+1):
        v_weights[i] = (v_weights[i]*beta) + ((del_W['W' + str(i+1)]**2) * (1-beta))
        v_biases[i] = (v_biases[i]*beta) + ((del_b['b' + str(i+1)]**2) * (1-beta))

        weights[i] = weights[i] - (((del_W['W' + str(i+1)]/np.sqrt(v_weights[i] + ep)))*eta)
        biases[i] = biases[i] - (((del_b['b' + str(i+1)]/np.sqrt(v_biases[i] + ep)))*eta)

        i+=1

      k+=1
    print(j)
    train_acc = train_accuracy(mini_batch_trainy,y_predicted,trainy)
    print("train_acc : ",train_acc*100)
    print("train_loss : ", tloss/number_batches)
    val_acc = test_accuracy(valX,valy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_acc : ",val_acc*100)
    vloss = val_loss(valX,valiy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_loss : ",vloss)

    train_loss_list.append(tloss/number_batches)
    val_loss_list.append(vloss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    j+=1 
  plot_lists = [train_loss_list,val_loss_list,train_acc_list,val_acc_list]
  return h[-1],weights,biases,plot_lists

def adam(trainX, trainy, number_hidden_layers = 1, hidden_layer_size = 4, eta = 0.1, initial_weights = 'random', activation_function = 'sigmoid', epochs = 1, output_function = 'softmax', mini_batch_size=4,loss_function = 'cross_entropy'):

#initialize layers of neural networks
  layers = []
  layer1 = {'input_size' : input_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
  layers.append(layer1)
  
  i=0
  while(i!=number_hidden_layers-1):
    hlayer = {'input_size' : hidden_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
    layers.append(hlayer)
    i+=1
  
  layern = {'input_size' : hidden_layer_size, 'output_size' : output_layer_size, 'function' : output_function}
  layers.append(layern)

#initialize weights and biases

  weights,biases =initialize_weights_and_biases(layers,number_hidden_layers,'random')

  number_batches = len(trainX)/mini_batch_size

  mini_batch_trainX = np.array(np.array_split(trainX, number_batches))
  mini_batch_trainy = np.array(np.array_split(trainy, number_batches))

  beta1 = 0.9
  beta2 = 0.999
  ep = 1e-3

  v_weights = []
  v_biases = []
  v_hat_weights = []
  v_hat_biases = []
  m_weights = []
  m_biases = []
  m_hat_weights = []
  m_hat_biases = [] 

  i = 0
  while(i!=number_hidden_layers+1):
    v_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    v_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    v_hat_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    v_hat_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    m_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    m_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    m_hat_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    m_hat_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    i+=1

  train_loss_list = []
  val_loss_list = []
  train_acc_list = []
  val_acc_list = []
  h=None
  j = 0
  c = 0
  while(j!=epochs):
    k=0
    tloss = 0
    vloss = 0
    y_predicted = []
    while(k!=number_batches):
      c+=1
      a,h = forward_propagation(mini_batch_trainX[k],weights,biases,number_hidden_layers, activation_function, output_function)
      y_predicted.append(h[-1])

      if loss_function == 'cross_entropy':
        tloss += cross_entropy(h[-1],mini_batch_trainy[k])
      elif loss_function == 'mse':
        tloss += mean_squared_error(h[-1],mini_batch_trainy[k])
      else:
        print('wrong loss function')
      
      del_W,del_b = backward_propagation(mini_batch_trainy[k],mini_batch_trainX[k],h[-1],a,h,weights,number_hidden_layers ,activation_function)

      i = 0
      while(i!=len(weights)):
        v_weights[i] = (v_weights[i]*beta2) + (((del_W['W'+str(i+1)])**2)*(1-beta2))
        v_biases[i] = (v_biases[i]*beta2) + (((del_b['b'+str(i+1)])**2)*(1-beta2))

        m_weights[i] = (m_weights[i]*beta1) + (del_W['W' + str(i+1)]*(1-beta1))
        m_biases[i] = (m_biases[i]*beta1) + (del_b['b' + str(i+1)]*(1-beta1))

        v_hat_weights[i] = (v_weights[i]/(1-beta2**c))
        v_hat_biases[i] = (v_biases[i]/(1-beta2**c))

        m_hat_weights[i] = (m_weights[i]/(1-beta1**c))
        m_hat_biases[i] = (m_biases[i]/(1-beta1**c))

        weights[i] = weights[i] - ((m_hat_weights[i]*eta/np.sqrt(v_hat_weights[i] + ep)))
        biases[i] = biases[i] - ((m_hat_biases[i]*eta/np.sqrt(v_hat_biases[i] + ep)))
        i+=1
      k+=1

    print(j)
    train_acc = train_accuracy(mini_batch_trainy,y_predicted,trainy)
    print("train_acc : ",train_acc*100)
    print("train_loss : ", tloss/number_batches)
    val_acc = test_accuracy(valX,valy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_acc : ",val_acc*100)
    vloss = val_loss(valX,valiy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_loss : ",vloss)

    train_loss_list.append(tloss/number_batches)
    val_loss_list.append(vloss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    j+=1

  plot_lists = [train_loss_list,val_loss_list,train_acc_list,val_acc_list]
  return h[-1],weights,biases,plot_lists

def nadam(trainX, trainy, number_hidden_layers = 1, hidden_layer_size = 4, eta = 0.1, initial_weights = 'random', activation_function = 'sigmoid', epochs = 1, output_function = 'softmax', mini_batch_size=4,loss_function = 'cross_entropy'):
  
#initialize layers of neural networks
  layers = []
  layer1 = {'input_size' : input_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
  layers.append(layer1)
  
  i=0
  while(i!=number_hidden_layers-1):
    hlayer = {'input_size' : hidden_layer_size, 'output_size' : hidden_layer_size, 'function' : activation_function}
    layers.append(hlayer)
    i+=1
  
  layern = {'input_size' : hidden_layer_size, 'output_size' : output_layer_size, 'function' : output_function}
  layers.append(layern)

#initialize weights and biases

  weights,biases =initialize_weights_and_biases(layers,number_hidden_layers,'random')

  number_batches = len(trainX)/mini_batch_size

  mini_batch_trainX = np.array(np.array_split(trainX, number_batches))
  mini_batch_trainy = np.array(np.array_split(trainy, number_batches))

  beta1 = 0.9
  beta2 = 0.999
  ep = 1e-3

  v_weights = []
  v_biases = []
  v_hat_weights = []
  v_hat_biases = []
  m_weights = []
  m_biases = []
  m_hat_weights = []
  m_hat_biases = [] 

  i = 0
  while(i!=number_hidden_layers+1):
    v_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    v_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    v_hat_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    v_hat_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    m_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    m_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    m_hat_weights.append(np.zeros((len(weights[i]),len(weights[i][0]))))
    m_hat_biases.append(np.zeros((len(biases[i]),len(biases[i][0]))))
    i+=1

  train_loss_list = []
  val_loss_list = []
  train_acc_list = []
  val_acc_list = []
  h=None
  j = 0
  c = 0
  while(j!=epochs):
    k=0
    tloss = 0
    vloss = 0
    y_predicted = []
    while(k!=number_batches):
      lookahead_weights = []
      lookahead_biases = []
      lookahead_v_hat_weights = []
      lookahead_v_hat_biases = []
      lookahead_m_hat_weights = []
      lookahead_m_hat_biases = []
      l = 0
      c+=1
      while(l!=number_hidden_layers+1):
        lookahead_v_hat_weights.append((v_weights[l]*beta2)/(1 - (beta2**c)))
        lookahead_v_hat_biases.append((v_biases[l]*beta2)/(1 - (beta2**c)))

        lookahead_m_hat_weights.append((m_weights[l]*beta1)/(1 - beta1**c))
        lookahead_m_hat_biases.append((m_biases[l]*beta1)/ (1 - beta1**c))

        lookahead_weights.append(weights[l] - (lookahead_m_hat_weights[l] / np.sqrt(lookahead_v_hat_weights[l] + ep))*eta)
        lookahead_biases.append(biases[l] - (lookahead_m_hat_biases[l] / np.sqrt(lookahead_v_hat_biases[l] + ep))*eta)
        l+=1

      a,h = forward_propagation(mini_batch_trainX[k],lookahead_weights,lookahead_biases,number_hidden_layers, activation_function, output_function)
      y_predicted.append(h[-1])

      if loss_function == 'cross_entropy':
        tloss += cross_entropy(h[-1],mini_batch_trainy[k])
      elif loss_function == 'mse':
        tloss += mean_squared_error(h[-1],mini_batch_trainy[k])
      else:
        print('wrong loss function')

      del_W,del_b = backward_propagation(mini_batch_trainy[k],mini_batch_trainX[k],h[-1],a,h,lookahead_weights,number_hidden_layers ,activation_function)

      i = 0
      while(i!=len(weights)):
        v_weights[i] = (v_weights[i]*beta2) + (((del_W['W'+str(i+1)])**2)*(1-beta2))
        v_biases[i] = (v_biases[i]*beta2) + (((del_b['b'+str(i+1)])**2)*(1-beta2))

        m_weights[i] = (m_weights[i]*beta1) + (del_W['W' + str(i+1)]*(1-beta1))
        m_biases[i] = (m_biases[i]*beta1) + (del_b['b' + str(i+1)]*(1-beta1))

        v_hat_weights[i] = (v_weights[i]/(1-beta2**c))
        v_hat_biases[i] = (v_biases[i]/(1-beta2**c))

        m_hat_weights[i] = (m_weights[i]/(1-beta1**c))
        m_hat_biases[i] = (m_biases[i]/(1-beta1**c))

        weights[i] = weights[i] - ((m_hat_weights[i]*eta/np.sqrt(v_hat_weights[i] + ep)))
        biases[i] = biases[i] - ((m_hat_biases[i]*eta/np.sqrt(v_hat_biases[i] + ep)))
        i+=1
      k+=1
    print(j)
    train_acc = train_accuracy(mini_batch_trainy,y_predicted,trainy)
    print("train_acc : ",train_acc*100)
    print("train_loss : ", tloss/number_batches)
    val_acc = test_accuracy(valX,valy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_acc : ",val_acc*100)
    vloss = val_loss(valX,valiy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("val_loss : ",vloss)

    train_loss_list.append(tloss/number_batches)
    val_loss_list.append(vloss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    j+=1

  plot_lists = [train_loss_list,val_loss_list,train_acc_list,val_acc_list]
  return h[-1],weights,biases,plot_lists

def train(trainX,trainy,textX,testy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,mini_batch_size,loss_function,optimizer):
  #num of hidden layers = 3
  #size of each hidden layer = 128
  #learning rate = 0.1
  #batch size = 32
  #activation = sigmoid
  #output = softmax
  #loss = cross entropy
  # print('Batch Gradient Descent')
  # print()
  # hL,weights,biases,plot_list = gradient_descent(trainX,trainy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,mini_batch_size,loss_function)
  # test_ac = test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function)
  # print("test_accuracy = ", test_ac*100,'%')
  # print(plot_list)

  if optimizer=='sgd':
    print()
    print('Stochastic Gradient Descent')
    print()
    hL,weights,biases,plot_list = gradient_descent(trainX,trainy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,1,loss_function)
    test_ac = test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("test_accuracy = ", test_ac*100,'%')
    print(plot_list)

  # sigmoid 3 128 0.001 best
  # tanh 3 128 32 0.001 best
  elif optimizer=='momentum':
    print()
    print('Momentum Based Gradient Descent')
    print()
    hL,weights,biases,plot_list = momentum_based_gradient_descent(trainX,trainy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,mini_batch_size,loss_function)
    test_ac = test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("test_accuracy = ", test_ac*100,'%')
    print(plot_list)

  # sigmoid 3 128 0.001 5 best
  # tanh 3 128 32 0.001 5 best
  elif optimizer=='nag':
    print()
    print('Nestrov')
    print()

    hL,weights,biases,plot_list = nestrov_accelerated_gradient_descent(trainX,trainy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,mini_batch_size,loss_function)
    test_ac = test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("test_accuracy = ", test_ac*100,'%')
    print(plot_list)

  # sigmoid 3 128 0.001 5 best
  # tanh 3 128 32 0.001 5 best
  elif optimizer=='rmsprop':
    print()
    print('RMSProp')
    print()

    hL,weights,biases,plot_list = rmsprop(trainX,trainy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,mini_batch_size,loss_function)
    test_ac = test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("test_accuracy = ", test_ac*100,'%')
    print(plot_list)

  elif optimizer == 'adam':
    print()
    print('Adam')
    print()

    hL,weights,biases,plot_list = adam(trainX,trainy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,mini_batch_size,loss_function)
    test_ac = test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("test_accuracy = ", test_ac*100,'%')
    print(plot_list)

  elif optimizer == 'nadam':
    print()
    print('NAdam')
    print()
    hL,weights,biases,plot_list = nadam(trainX,trainy,number_hidden_layers,hidden_layer_size,eta,init_type,activation_function,epochs,output_function,mini_batch_size,loss_function)
    test_ac = test_accuracy(testX,testy,weights,biases,number_hidden_layers,activation_function,output_function)
    print("test_accuracy = ", test_ac*100,'%')
    print(plot_list)

train(trainX,trainy,testX,needed_y_test,3,128,1e-3,'xavier','tanh',15,'softmax',30,'cross_entropy','nadam')
