import theano as th
import theano.tensor as te

x = te.vector('x')
y = te.exp(x-te.max(x))
y = y/te.sum(y)
softmax = th.function([x], [y])

print softmax([1, 0])
