import torch
from copy import deepcopy
import torch.distributions as dist

#--------------------------Useful functions and OPS ---------------------------#
class Normal(dist.Normal):

    def __init__(self, loc, scale):

        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()


        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """

        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]

        return Normal(*ps)

    def log_prob(self, x):

        self.scale = torch.nn.functional.softplus(self.optim_scale)

        return super().log_prob(x)

def _hashmap(*x):
    x = x[1:]
    # List (key, value, key, value, ....)
    return_x = {}
    if len(x)%2 == 0:
        for i in range(0, len(x)-1, 2):
            if torch.is_tensor(x[i]):
                key_value  = x[i].item()
            else:
                key_value  = x[i]
            value = x[i+1]
            try:
                if not torch.is_tensor(value):
                    if isinstance(value, list):
                        value = torch.tensor(value, dtype=x.dtype)
                    else:
                        value = torch.tensor([value], dtype=x.dtype)
                return_x[key_value] = value
            except:
                return_x[key_value] = value
    else:
        raise IndexError('Un-even key-value pairs')

    return return_x

def _totensor(*x):
    dtype=torch.float32 # Hard coding to float for now
    if not torch.is_tensor(x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=dtype)
        else:
            x = torch.tensor([x], dtype=dtype)
    return x

def _vector(*x):
    x = x[1:]
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    else:
        # Maybe single value
        return torch.tensor([x], dtype=torch.float32)

def _put(addr, x, idx_or_key, value):
    x_= deepcopy(x)
    if isinstance(x_, dict):
        if torch.is_tensor(idx_or_key):
            idx_or_key  = idx_or_key.item()
        try:
            if not torch.is_tensor(value):
                value = _vector(x=x)
            x_[idx_or_key] = value
        except:
            raise IndexError('Key {} cannot put in the dict'.format(idx_or_key))
        return x_
    elif isinstance(x_, list):
        try:
            x_[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x_
    elif torch.is_tensor(x_):
        try:
            try:
                if idx_or_key.type() == 'torch.FloatTensor':
                    idx_or_key = idx_or_key.type(torch.LongTensor)
            except:
                idx_or_key = torch.tensor(idx_or_key, dtype=torch.long)
            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=x.dtype)
            if len(x_.size()) > 1:
                try:
                    x_[0, idx_or_key] = value
                except:
                    x_[idx_or_key] = value
            else:
                x_[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x_
    else:
         raise AssertionError('Unsupported data structure')

def _remove(addr, x, idx_or_key):
    if isinstance(x, dict):
        try:
            if isinstance(idx_or_key, float):
                idx_or_key = int(idx_or_key)
            x.pop(idx_or_key, None)
        except:
            raise IndexError('Key {} is not present in the dict'.format(idx_or_key))
        return x
    elif isinstance(x, list):
        try:
            if isinstance(idx_or_key, float):
                idx_or_key = int(idx_or_key)
            x.pop(idx_or_key)
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))
        return x
    elif torch.is_tensor(x):
        try:
            x = torch.cat((x[:idx_or_key], x[(idx_or_key+1):]))
        except:
            raise IndexError('Index {} is not present in the tensor'.format(idx_or_key))

        return x
    else:
         raise AssertionError('Unsupported data structure')

def _append(addr, x, value):
    if isinstance(x, list):
        if isinstance(value, list):
            x.extend(value)
        else:
            # single value
            x.append(value)
    elif torch.is_tensor(x):
        if not torch.is_tensor(value):
            if isinstance(value, list):
                value = torch.tensor(value, dtype=x.dtype)
            else:
                value = torch.tensor([value], dtype=x.dtype)
        if x.dtype != value.dtype:
            value = value.type(x.dtype)
        try:
            if len(x.size()) > 1 and (not(x.shape[0] == 1 and x.shape[1] == 1)):
                x = torch.squeeze(x)
            elif len(x.size()) > 1 and (x.shape[0] == 1 and x.shape[1] == 1):
                x = torch.squeeze(x, dim=1)
            x = torch.cat((x, value))
        except Exception as e:
            print(x.dtype)
            print(value.dtype)
            raise AssertionError('Cannot append the torch tensors, due to: ', str(e))
        return x
    else:
        raise AssertionError('Unsupported data structure')

def _get(addr, x, idx):
    if isinstance(x, dict):
        if torch.is_tensor(idx):
            idx = idx.item()
        else:
            idx = idx
        return x[idx]
    elif isinstance(x, list):
        if isinstance(idx, float):
            idx = int(idx)
        return x[idx]
    elif torch.is_tensor(x):
        try:
            if idx.type() == 'torch.FloatTensor':
                idx = idx.type(torch.LongTensor)
        except:
            idx = torch.tensor(idx, dtype=torch.long)
        if len(x.size()) > 1:
            x = torch.squeeze(x)
        return x[idx]
    else:
        raise AssertionError('Unsupported data structure')

def _squareroot(addr, x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(x)

def _add(addr, a, b):
    return torch.add(a, b)

def _sub(addr, a, b):
    return torch.sub(a, b)

def _mul(addr, a, b):
    return torch.mul(a, b)

def _div(addr, a, b):
    return torch.div(a, b)

def _first(addr, x):
    if isinstance(x, list):
        return x[0]
    elif torch.is_tensor(x):
        if len(x.size()) > 1 and (not(x.shape[0] == 1 and x.shape[1] == 1)):
            x = torch.squeeze(x)
        return x[0]

def _second(addr, x):
    if isinstance(x, list):
        return x[1]
    elif torch.is_tensor(x):
        if len(x.size()) > 1 and (not(x.shape[0] == 1 and x.shape[1] == 1)):
            x = torch.squeeze(x)
        return x[1]

def _last(addr, x):
    #import pdb; pdb.set_trace()
    if isinstance(x, list):
        return x[-1]
    elif torch.is_tensor(x):
        if len(x.size()) > 1 and (not(x.shape[0] == 1 and x.shape[1] == 1)):
            x = torch.squeeze(x)
        return x[-1]

def _rest(addr, x):
    if isinstance(x, list):
        return x[1:]
    elif torch.is_tensor(x):
        if len(x.size()) > 1 and (not(x.shape[0] == 1 and x.shape[1] == 1)):
            x = torch.squeeze(x)
        return x[1:]

def _lessthan(addr, a, b):
    return torch.Tensor([a < b])

def _greaterthan(addr, a, b):
    return torch.Tensor([a > b])

def _equals(addr, a, b):
    return torch.Tensor([a == b])

def _greaterthanequal(addr, a, b):
    return torch.Tensor([a >= b])

def _lessthanequal(addr, a, b):
    return torch.Tensor([a <= b])

def _and(addr, a, b):
    return torch.Tensor([a and b])

def _or(addr, a, b):
    return torch.Tensor([a or b])

def _log(addr, x):
    return torch.log(x)

def _normal(addr, mu, sigma):
    return Normal(mu, sigma)

def _beta(addr, a, b):
    return dist.beta.Beta(concentration1=a, concentration0=b)

def _gamma(addr, concentration, rate):
    return dist.gamma.Gamma(concentration=concentration, rate=rate)

def _uniform(addr, low, high):
    return dist.uniform.Uniform(low=low, high=high)

def _exponential(addr, rate):
    return dist.exponential.Exponential(rate=rate)

def _discrete(addr, probs):
    return dist.categorical.Categorical(probs=probs)

def _dirichlet(addr, concentration):
    return dist.dirichlet.Dirichlet(concentration=concentration)

def _beroulli(addr, probs):
    return dist.bernoulli.Bernoulli(probs=probs)

def _empty(addr, x):
    return torch.Tensor([len(x) == 0])

def _cons(addr, x1, x2):
    if not torch.is_tensor(x1):
        if isinstance(x1, list):
            x1 = torch.tensor(x1, dtype=x.dtype)
        else:
            x1 = torch.tensor([x1], dtype=x.dtype)
    if not torch.is_tensor(x2):
        if isinstance(x2, list):
            x2 = torch.tensor(x2, dtype=x.dtype)
        else:
            x2 = torch.tensor([x2], dtype=x.dtype)
    try:
        if len(x1.size()) > 1:
            x1 = torch.squeeze(x1)
        if len(x2.size()) > 1:
            x2 = torch.squeeze(x2)
        value = torch.cat((x1, x2))
    except:
        raise AssertionError('Cannot append the torch tensors')

    return value

def push_addr(alpha, value):
    return alpha + value

# Primitives
env = {
       'push-address': push_addr,
       'empty?':       _empty,
       '+':            _add,
       '-':            _sub,
       '*':            _mul,
       '/':            _div,
       'vector':       _vector,
       'hash-map':     _hashmap,
       'first':        _first,
       'second':       _second,
       'last':         _last,
       'peek':         _last,
       'rest':         _rest,
       'get':          _get,
       'nth':          _get,
       'append':       _append,
       'remove':       _remove,
       'cons':         _cons,
       'conj':         _append,
       'put':          _put,
       '<':            _lessthan,
       '>':            _greaterthan,
       '=':            _equals,
       '>=':           _greaterthanequal,
       '<=':           _lessthanequal,
       'or':           _or,
       'and':          _and,
       'log':          _log,
       'sqrt':         _squareroot,
       'normal':       _normal,
       "beta":         _beta,
       "gamma":        _gamma,
       "uniform":      _uniform,
       "exponential":  _exponential,
       "discrete":     _discrete,
       "dirichlet":    _dirichlet,
       "bernoulli":    _beroulli,
       "flip":         _beroulli,
       "uniform-continuous": _uniform
      }
