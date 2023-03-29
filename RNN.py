import numpy as np 

def softmax(self, indata):
        indata -= np.max(indata)
        outdata = np.exp(indata)
        outdata = outdata / np.sum(outdata)
        return outdata

class RNN:

    def __init__(self, hidden_size, seq_length, vocab_size, data_size, stoi, itos):
        # data I/O
        self.data_size = data_size
        self.stoi = stoi # mapping from char to index
        self.itos = itos # mapping from index to char

        # hyperparameters
        self.hidden_size = hidden_size # size of hidden state vector
        self.seq_length = seq_length # max length to backprop 
        self.vocab_size = vocab_size # dimension of input and output vectors

        # initialise trainable parameters
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

        # loss record
        self.loss = 0

    def train(self, data, total_iteration, lr, show_loss=True):
        # index for loading data
        load_idx = 0
        # previous hidden state
        hprev = np.zeros((self.hidden_size, 1))
        # RMSprop decay rate and cache
        self.decay_rate = 0.99
        self.mWhh = np.zeros_like(self.Whh)
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        # total iteration count 
        iteration = 0

        while True:
            # reset the loading index and hprev 
            if load_idx + self.seq_length + 1 >= self.data_size:
                load_idx = 0
                hprev = np.zeros((self.hidden_size, 1))

            # inputs and targets in this iteration 
            inputs = [self.stoi[ch] for ch in data[load_idx:load_idx+self.seq_length]]
            targets = [self.stoi[ch] for ch in data[load_idx+1:load_idx+self.seq_length+1]]

            # forward-backward-update
            hprev = self.train_one_loop(inputs, targets, hprev,lr=lr)
            
            # print loss
            if show_loss == True and iteration % 10 == 0:
                print(f'Iteration : {iteration} | Loss : {self.loss}')

            # update loading index and iteration counter 
            load_idx += self.seq_length
            iteration += 1

            if iteration > total_iteration:
                break 

    # [forward -> backward -> update] for one time 
    def train_one_loop(self, inputs, targets, hprev, lr):
        # forward pass
        # -----------------------------------------------------------------------------
        hs, xs, ys, ps = {}, {}, {}, {}
        loss = 0
        hs[-1] = np.copy(hprev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = self.softmax(ys[t])
            loss += -np.log(ps[t][targets[t]]).item()
        self.loss = loss 

        # backward pass
        # -----------------------------------------------------------------------------
        # gradients 
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhprev = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(inputs))):
            dy = ps[t]
            dy[targets[t]] -= 1
            dby += dy
            dWhy += dy @ hs[t].T
            dh = self.Why.T @ dy + dhprev
            dh_before_tanh = dh * (1 - hs[t]**2)
            dbh += dh_before_tanh
            dWxh += dh_before_tanh @ xs[t].T
            dWhh += dh_before_tanh @ hs[t-1].T
            dhprev = self.Whh.T @ dh_before_tanh

        # gradient clipping 
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -10, 10, out=dparam)

        # update: RMSprop
        # -----------------------------------------------------------------------------
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                 [dWxh, dWhh, dWhy, dbh, dby],
                                 [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem = self.decay_rate * mem + (1-self.decay_rate) * dparam*dparam
            param += -lr * dparam / (np.sqrt(mem)+ 1e-8)

        return hs[len(inputs)-1]

    # generate some results 
    def generate(self, samples):
        h = np.zeros((self.hidden_size, 1))
        x = np.zeros((self.vocab_size, 1))
        x[0] = 1
        result_idx = []

        for _ in range(samples):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = self.softmax(y)
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            result_idx.append(idx)
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

        result_char = [self.itos[i] for i in result_idx]
        result = ''.join(result_char)
        
        print(result)
