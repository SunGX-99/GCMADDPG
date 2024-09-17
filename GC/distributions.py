import tensorflow as tf
import numpy as np
import tf_util as U
from tensorflow.python.ops import math_ops
from multi_discrete import MultiDiscrete
from tensorflow.python.ops import nn
from gym import spaces

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return SoftCategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return [self.ncat]
    def sample_dtype(self):
        return tf.float32

class MultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return tf.int32

class SoftMultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return SoftMultiCategoricalPd
    def pdfromflat(self, flat):
        return SoftMultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [sum(self.ncats)]
    def sample_dtype(self):
        return tf.float32

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd
    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return BernoulliPd
    def param_shape(self):
        return [self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.int32

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return U.argmax(self.logits, axis=1)
    def logp(self, x):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)
    def sample(self):
        u = tf.random.uniform(tf.shape(self.logits))
        return U.argmax(self.logits - tf.log(-tf.log(u)), axis=1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftCategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return U.softmax(self.logits, axis=-1)
    def logp(self, x):
        return -tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)
    def sample(self):
        u = tf.random.uniform(tf.shape(self.logits))
        return U.softmax(self.logits - tf.log(-tf.log(u)), axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(Pd):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = tf.constant(low, dtype=tf.int32)
        self.categoricals = list(map(CategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.low + tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
    def logp(self, x):
        return tf.add_n([p.logp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        return self.low + tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftMultiCategoricalPd(Pd):  # doesn't work yet
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = tf.constant(low, dtype=tf.float32)
        self.categoricals = list(map(SoftCategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].mode())
        return x
    def logp(self, x):
        x = tf.stack([p.logp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
        return tf.reduce_sum(x, axis=1)
    def kl(self, other):
        return tf.reduce_sum([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)], axis=1)
    def entropy(self):
        return tf.reduce_sum([p.entropy() for p in self.categoricals], axis=1)
    def sample(self):
        return tf.stack([self.low[i] + self.categoricals[i].sample() for i in range(len(self.categoricals))])
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.mean = flat[:, :flat.shape[1] // 2]
        self.logstd = flat[:, flat.shape[1] // 2:]
    def flatparam(self):
        return tf.concat([self.mean, self.logstd], axis=1)
    def mode(self):
        return self.mean
    def logp(self, x):
        log2pi = np.log(2 * np.pi)
        logstd = self.logstd
        return -0.5 * tf.reduce_sum(log2pi + 2 * logstd + tf.square(x - self.mean) / tf.exp(2 * logstd), axis=-1)
    def kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.logstd - other.logstd) + tf.exp(2 * other.logstd) - tf.exp(2 * self.logstd)) / (2 * tf.exp(2 * other.logstd)), axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + 0.5 * np.log(2 * np.pi * np.e * tf.exp(2 * self.logstd)), axis=-1)
    def sample(self):
        return self.mean + tf.exp(self.logstd / 2) * tf.random.normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BernoulliPd(Pd):
    def __init__(self, flat):
        self.probs = tf.sigmoid(flat)
    def flatparam(self):
        return self.probs
    def mode(self):
        return tf.cast(self.probs > 0.5, tf.int32)
    def logp(self, x):
        return x * tf.math.log(self.probs + 1e-8) + (1 - x) * tf.math.log(1 - self.probs + 1e-8)
    def kl(self, other):
        return tf.reduce_sum(self.probs * tf.math.log(self.probs / (other.probs + 1e-8)) + (1 - self.probs) * tf.math.log((1 - self.probs) / (1 - other.probs + 1e-8)), axis=-1)
    def entropy(self):
        return -tf.reduce_sum(self.probs * tf.math.log(self.probs + 1e-8) + (1 - self.probs) * tf.math.log(1 - self.probs + 1e-8), axis=-1)
    def sample(self):
        return tf.cast(tf.random.uniform(tf.shape(self.probs)) < self.probs, tf.int32)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    if isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.low, ac_space.high)
    elif isinstance(ac_space, spaces.Box):
        if ac_space.shape[0] == 1:
            return DiagGaussianPdType(ac_space.shape[0])
        elif ac_space.shape[0] > 1:
            return SoftCategoricalPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.shape[0])
    else:
        raise NotImplementedError
