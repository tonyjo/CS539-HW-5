import torch
import torch.distributions as dist

#--------------------------Useful functions and OPS ---------------------------#
class Normal(dist.Normal):

    def __init__(self, alpha, loc, scale):

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


def _hashmap(x):
    # List (key, value, key, value, ....)
    return_x = {}
    if len(x)%2 == 0:
        for i in range(0, len(x)-1, 2):
            if torch.is_tensor(x[i]):
                key_value  = x[i].item()
            else:
                key_value  = x[i]
            value = x[i+1]
            if not torch.is_tensor(value):
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=x.dtype)
                else:
                    value = torch.tensor([value], dtype=x.dtype)
            return_x[key_value] = value
    else:
        raise IndexError('Un-even key-value pairs')

    return return_x


def _vector(x):
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    else:
        # Maybe single value
        return torch.tensor([x], dtype=torch.float32)


def _put(x, idx_or_key, value):
    if isinstance(x, dict):
        if torch.is_tensor(idx_or_key):
            idx_or_key  = idx_or_key.item()
        try:
            if not torch.is_tensor(value):
                value = _vector(x=x)
            x[idx_or_key] = value
        except:
            raise IndexError('Key {} cannot put in the dict'.format(idx_or_key))
        return x
    elif isinstance(x, list):
        try:
            x[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x
    elif torch.is_tensor(x):
        try:
            try:
                if idx_or_key.type() == 'torch.FloatTensor':
                    idx_or_key = idx_or_key.type(torch.LongTensor)
            except:
                idx_or_key = torch.tensor(idx_or_key, dtype=torch.long)
            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=x.dtype)
            if len(x.size()) > 1:
                try:
                    x[0, idx_or_key] = value
                except:
                    x[idx_or_key] = value
            else:
                x[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x
    else:
         raise AssertionError('Unsupported data structure')


def _remove(x, idx_or_key):
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


def _append(x, value):
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
        try:
            if len(x.size()) > 1:
                x = torch.squeeze(x)
            x = torch.cat((x, value))
        except:
            raise AssertionError('Cannot append the torch tensors')
        return x
    else:
        raise AssertionError('Unsupported data structure')


def _get(x, idx):
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


def _squareroot(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(x)


def _totensor(x, dtype=torch.float32):
    if not torch.is_tensor(x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=dtype)
        else:
            x = torch.tensor([x], dtype=dtype)
    return x

def _first(x):
    if isinstance(x, list):
        return x[0]
    elif torch.is_tensor(x):
        if len(x.size()) > 1:
            x = torch.squeeze(x)
        return x[0]

def _second(x):
    if isinstance(x, list):
        return x[1]
    elif torch.is_tensor(x):
        if len(x.size()) > 1:
            x = torch.squeeze(x)
        return x[1]

def _last(x):
    if isinstance(x, list):
        return x[-1]
    elif torch.is_tensor(x):
        if len(x.size()) > 1:
            x = torch.squeeze(x)
        return x[-1]


def push_addr(alpha, value):
    return alpha + value


# Primitives
env = {
       'normal':       Normal,
       'push-address': push_addr,
       '+':            torch.add,
       '-':            torch.sub,
       '*':            torch.mul,
       '/':            torch.div,
       'vector':       lambda *x: _vector(x),
       'hash-map':     lambda *x: _hashmap(x),
       'first':        lambda x: _first(x),
       'second':       lambda x: _second(x),
       'last':         lambda x: _last(x),
       'rest':         lambda x: x[1:],
       'get':          lambda x, idx: _get(x, idx),
       'append':       lambda x, y: _append(x, y),
       'remove':       lambda x, idx: _remove(x, idx),
       'put':          lambda x, idx, value: _put(x, idx, value),
       '<':            lambda a, b: a < b,
       '>':            lambda a, b: a > b,
       '=':            lambda a, b: a == b,
       '>=':           lambda a, b: a >= b,
       '<=':           lambda a, b: a <= b,
       'or':           lambda a, b: a or b,
       'and':          lambda a, b: a and b,
       'sqrt':         lambda x: _squareroot(x)
      }
