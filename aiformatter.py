#!/usr/bin/python3

import clang3.cindex
from clang3.cindex import Index
from clang3.cindex import conf
from clang3.cindex import TokenGroup
from ailearner import RNN

from sys import argv


class Formatter:
	def __init__(self, model='model.dat'):
		if not conf.loaded:
			conf.set_library_path("/usr/lib/llvm-3.8/lib")
		self.rnn = RNN.load(model) #[0-127] 128 total ascii characters, all non-ascii will be mapped onto 0
		self.modelfile = model

	def tokenize(self, filepath):
		exclude_decls_from_pch = 1
		display_diagnostics = 0

		index = Index.create(exclude_decls_from_pch)
		if not index:
			print("Can't create clang index")
			return None

		tu = index.parse(filepath.encode('utf-8'))
		return tu

	def load_original(self, filepath):
		original = ''
		with open(filepath, 'r') as f:
			for line in f:
				original += line

		return original

	def writeout_tokens(self, tu, original, filepath, debug = False):
		if not tu:
			print('tu is empty!!')
			return None

		self.rnn.reset_prediction()
		cursor = tu.cursor
		rng = cursor.extent
		with open(filepath, 'wb') as f:
			last_offset  = 0
			next_char_prediction = 0
			last_token = None
			last_char = ''
			for token in TokenGroup.get_tokens(tu, rng):
				out = ''
				inp = last_char
				original_white_spaces = ''
				predicted_white_spaces = ''

				if debug:
					out = str('tok = ' + str(token.spelling) +
						 " -> [" + str(token.extent.start.line) + ': ' + str(token.extent.start.column) + ', ' + str(token.extent.end.column) + "]" +
						" at {" + str(token.extent.start.offset) + ", " + str(token.extent.end.offset) + "}\n")

					f.write(out.encode('utf-8'))
					continue

				if last_offset != token.extent.start.offset:
					# white spaces before current token
					original_white_spaces = original[last_offset:token.extent.start.offset]

				# forming input sequence
				inp += original_white_spaces
				inp += token.spelling.decode('utf-8')

				# inserting predicted white spaces
				for ch in inp[:-1]:
					next_char_prediction = self.rnn.predict_char(ch)
					if next_char_prediction.isspace():
						predicted_white_spaces += next_char_prediction
					else:
						break
				last_char = inp[-1]

				# forming output sequence
				out += predicted_white_spaces
				out += token.spelling.decode('utf-8')
				# write it out
				f.write(out.encode('utf-8'))
				# next iteration prepare
				last_token = token
				last_offset = token.extent.end.offset

			# write out last CR
			f.write('\n'.encode('utf-8'))

	def get_model_number(self):
		import re
		return int(re.findall('\d+', self.modelfile)[-1])

	def get_output_file(self, filename):
		import os
		filename, ext = os.path.splitext(filename)
		model_number = self.get_model_number()
		return '%s.%d%s' % (filename, model_number, ext)

	def format_file(self, filepath, predict = None):
		tu = self.tokenize(filepath)
		if not tu:
			print("Can't parse source file " + filepath)
			return False

		original = self.load_original(filepath)
		out_filepath = self.get_output_file(filepath)
		self.writeout_tokens(tu, original, out_filepath)
		return True

def usage():
	print("Usage: " + argv[0] + " <source file> [model.dat]")

def main():
	if len(argv) != 2 and len(argv) != 3:
		usage()
		exit(-1)

	model = 'model.dat'
	if len(argv) == 3:
		model = argv[2]

	conf.set_library_path("/usr/lib/llvm-3.8/lib")
	fmt = Formatter(model)
	if not fmt.rnn:
		print("can't load model.dat!!! Please generate one using ailearner.py")
		exit(-1)

	fmt.format_file(argv[1])

if __name__ == '__main__':
	main()
