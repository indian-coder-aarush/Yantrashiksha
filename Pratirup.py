from Tanitra import length,mean,square,to_cons

class AnukramikPratirup:

    def __init__(self,layers = None):
        self.layers = []
        if not layers is None:
            for i in layers:
                self.add(i)

    def add(self,layer):
        self.layers.append(layer)

    def estimate(self,X):
        activation = X
        for i in self.layers:
            activation = i.forward(activation)
        return activation

    def learn(self,X,y,optimizer = 'Gradient Descent',epochs = 1000,lr = 0.01,tol = 0.00001):
        if optimizer == 'Gradient Descent':
            loss = 0
            for _ in range(epochs):
                prev_loss = loss
                loss = 0
                for i in range(length(X)):
                    y_pred = self.estimate(X[i])
                    loss += to_cons(mean(square(y_pred - y[i])) / length(X) )
                    req_loss = square(y_pred - y[i])/ length(X)
                    req_loss.backward()
                    for j in self.layers:
                        for k in j.params:
                            j.params[k].data = j.params[k].data - j.params[k].grad*lr/length(X)
                if prev_loss - loss < tol and _ != 0:
                    break
                if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                    raise RuntimeWarning("Error did not converge. Please consider increasing number of epochs.")
                print("Epoch no. ",_)
                print("  loss is ",loss)
                print(" ")