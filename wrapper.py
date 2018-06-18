from __future__ import absolute_import

import inspect


def set_f_args(f, **default_args):
	"""Set the default argument values of f and return
	   a corresponding function.
	"""
	def _set_f_default_args(f, args, default_args, kwargs):
		if inspect.isclass(f):
			# remove the self.
			num_args = f.__init__.func_code.co_argcount - 1
			arg_names = f.__init__.func_code.co_varnames[1:num_args]
		else:
			num_args = f.func_code.co_argcount
			arg_names = f.func_code.co_varnames[:num_args]
		
		cursor = 0
		merged_args = []
		for arg in arg_names:
			if arg == 'self':
				continue
			if arg in default_args:
				merged_args.append(default_args[arg])
			elif arg in kwargs:
				merged_args.append(kwargs[arg])
			else:
				merged_args.append(args[cursor])
				cursor += 1
		merged_args += list(args[cursor:])
		return f(*merged_args)
	
	# Since types.LambdaType is types.FunctionType,
	#   we have to distinguish the functions in a hacking style.
	if f.__name__ == '<lambda>':
		g = lambda *args, **kwargs:              \
			f(*args, **dict(default_args.items() \
			+ kwargs.items()))
	else:
		g = lambda *args, **kwargs: \
			_set_f_default_args(f, args, default_args, kwargs)
	
	return g
