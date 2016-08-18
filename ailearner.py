#!/usr/bin/python3

import numpy as np
from datetime import datetime
import sys
from sys import argv

def softmax(x):
#	print('softmax x dim:' + str(x.shape))
	exp_scores = np.exp(x)
	probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
	return probs


class RNN:
	def __init__(self, vocab_dim, hidden_dim=100):
		self.vocab_dim = vocab_dim
		self.hidden_dim = hidden_dim

		# svaed hidden state
		self.hprev = np.zeros((hidden_dim, 1))

		# input matrix
		self.U = np.random.randn(hidden_dim, vocab_dim) * 0.01
		# output matrix
		self.V = np.random.randn(vocab_dim, hidden_dim) * 0.01
		# transition matrix
		self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
		# hidden bias
		self.bh = np.zeros((hidden_dim, 1))
		# output bias
		self.by = np.zeros((vocab_dim, 1))

		# memory for adaptive gradient
		self.mU = np.zeros_like(self.U)
		self.mV = np.zeros_like(self.V)
		self.mW = np.zeros_like(self.W)
		self.mbh = np.zeros_like(self.bh)
		self.mby = np.zeros_like(self.by)

		# total loss
		self.sequence_len = 25
		self.loss = 0

		self.ch_to_x = {}
		for i in range(self.vocab_dim):
			ch = chr(i)
			self.ch_to_x[ch] = self.convert_ch_to_x(ch)

	def reset_epoch(self):
		self.loss = -np.log(1.0 / self.vocab_dim) * self.sequence_len
		self.hprev = np.zeros_like(self.hprev)

	def reset_prediction(self):
		self.hprev = np.zeros_like(self.hprev)

	def convert_ch_to_x(self, ch):
		x = np.zeros((self.vocab_dim, 1))
		x[ord(ch)][0] = 1
		return x

	def get_data(self, string):
		return np.array([self.ch_to_x[ch] for ch in string])

	def replace_non_ascii(self, string):
		return ''.join([ch if ord(ch) < 128 else '\x01' for ch in string])

	def forward_propagation(self, x, y_chars = None):
		# total number of input samples
		T = len(x)

		# saved hidden states T times
		s = np.zeros((T, self.hidden_dim, 1))
		# and last one row is hprev
		s[-1] = np.copy(self.hprev)

		# saved previous outputs T times
		o = np.zeros((T, self.vocab_dim, 1)) # 1-to-vocab_size representation

		#print('T=' + str(T))
		#print('s' + str(s.shape))
		#print('x' + str(x.shape))
		#print('U' + str(self.U.shape))
		#print('W' + str(self.W.shape))
		#print('V' + str(self.V.shape))
		#print('U*x' + str(self.U.dot(x[0]).shape))
		#print('W*s' + str(self.W.dot(s[0]).shape))
		#print('bh' + str(self.bh.shape))

		# for each char
		if y_chars:
			#print('T=' + str(T))
			for t in range(T):
				s[t] = np.tanh(self.U.dot(x[t]) + self.W.dot(s[t-1]) + self.bh)
				o[t] = softmax(self.V.dot(s[t]) + self.by)
				self.loss += -np.log(o[t][ord(y_chars[t])])
		else:
			for t in range(T):
				s[t] = np.tanh(self.U.dot(x[t]) + self.W.dot(s[t-1]) + self.bh)
				o[t] = softmax(self.V.dot(s[t]) + self.by)

		self.hprev = np.copy(s[-1])

		return [o, s, self.loss]

	def predict(self, x):
		o, s, loss = self.forward_propagation(x)

		#print(o[len(x)-1,:])
		return np.argmax(o[len(x)-1:])
		#return o[len(x)-1,:] # select only last state from o

	def predict_char(self, ch):
		return chr(self.predict(np.array([self.ch_to_x[ch]])))

	def back_propagation(self, x, y, y_chars):
		T = len(y)

		# forward prop step
		o, s, loss = self.forward_propagation(x, y_chars)
		
		# gradients
		dLdU = np.zeros_like(self.U)
		dLdV = np.zeros_like(self.V)
		dLdW = np.zeros_like(self.W)
		dLdbh = np.zeros_like(self.bh)
		dLdby = np.zeros_like(self.by)

		dhnext = np.zeros_like(s[0])

		# calculate errors [vectorized]
		delta_o = o - y
		#print('delta_o=' + str(delta_o.shape))
		# for each output backwards
		for t in reversed(range(T)):
			dLdV += np.outer(delta_o[t], s[t].T)
			dLdby += delta_o[t]

			#initial dh calculation
			dh = self.V.T.dot(delta_o[t]) + dhnext
			dhraw = (1 - (s[t] ** 2)) * dh
			dLdbh += dhraw

			dLdW += np.outer(dhraw, s[t-1])
			dLdU += np.outer(dhraw, x[t])
			
			# update delta for next step
			dhnext = self.W.T.dot(dhraw)

		for dparam in [dLdU, dLdV, dLdW, dLdbh, dLdby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradient
		return [dLdU, dLdV, dLdW, dLdbh, dLdby]

	# adaptive gradient learning step
	def adagrad_step(self, x, y, y_chars, learning_rate):
		dLdU, dLdV, dLdW, dLdbh, dLdby = self.back_propagation(x, y, y_chars)
		# Change parameters according to gradients and learning rate
		for param, dparam, mem in zip(
								[self.U, self.V, self.W, self.bh, self.by     ],
								[dLdU,    dLdV,    dLdW,    dLdbh,    dLdby   ],
								[self.mU, self.mV, self.mW, self.mbh, self.mby]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) #adagrad update

	@staticmethod
	def load(filename):
		import pickle
		from os import path
		if path.exists(filename):
			with open(filename, 'rb') as f:
				return pickle.load(f)
		return None

	def save(self, filename):
		import pickle
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
			return True

		return False

	# Outer AdaGrad Loop
	# - self: The RNN model instance
	# - filename: text code file
	# - learning_rate: Initial learning rate for SGD
	# - nepoch: Number of times to iterate through the complete dataset
	# - evaluate_loss_after: Evaluate the loss after this many epochs
	def train(self, filename, learning_rate=0.1, nepoch=100, evaluate_loss_after=3):
		# We keep track of the losses so we can plot them later
		losses = []
		num_examples_seen = 0
		string = ''
		with open(filename, 'rb') as f:
			bytes = f.read()
			print('Data have been read from %s, size=%d' % (filename, len(bytes)))
			string = self.replace_non_ascii(bytes.decode('866'))
			print('Data have been decoded into strin')

		for epoch in range(nepoch):
			# Optionally evaluate the loss
			if (epoch % evaluate_loss_after == 0):
				losses.append((num_examples_seen, self.loss))
				time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				print( "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, self.loss))
				# Adjust the learning rate if loss increases
				if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
					learning_rate = learning_rate * 0.5 
					print( "Setting learning rate to %f" % learning_rate)
				self.save('model.' + str(epoch) + ".dat")
				sys.stdout.flush()

			self.reset_epoch()
			print('epoch.' +str(epoch))
			beg = 0
			end = beg + self.sequence_len
			while(end < len(string)):
				# input sequence of enoded vectors
				X_seq = self.get_data(string[beg:end])
				# true output chars for the above input is shifted by one char the same sequence
				y_chars = string[beg+1:end+1]
				# encoded output
				y_seq = self.get_data(y_chars)
				# One adagrad step
				self.adagrad_step(X_seq, y_seq, y_chars, learning_rate)
				num_examples_seen += 1
				# iterate
				beg += self.sequence_len
				end = beg + self.sequence_len

def usage():
	print("Usage: " + argv[0] + " <samples> [APPEND]")

def main():
	if len(argv) != 2 and len(argv) != 3:
		usage()
		exit(-1)

	append_model = (len(argv) == 3)
	filename = argv[1]

	model = RNN(128) #[0-127] 128 total ascii characters, all non-ascii will be mapped onto 0

	loaded_model = model.load('model.dat')
	if not loaded_model:
		model.train(filename)
		model.save('model.dat')
	elif append_model:
		loaded_model.train(filename)
		loaded_model.save('model.dat')
	else:
		print('model.dat already presented!!!! Remove it if you want to regenerate model')
		exit(-1)

if __name__ == '__main__':
	main()
