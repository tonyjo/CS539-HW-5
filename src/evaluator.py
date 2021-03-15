import json
import torch
from daphne import daphne
from pyrsistent import pmap, plist
# Tests
from tests import is_tol, run_prob_test, load_truth
# Global env
from primitives import env as penv
from primitives import _totensor

def standard_env():
    "An environment with some Scheme standard procedures."
    # env = pmap(penv)
    # env = env.update({'alpha' : []})
    # return env

    penv.update({'alpha' : []})
    return penv

global_env = standard_env()

def evaluate(exp, sig={}, env=global_env):
    if env is None:
        env = standard_env()
    # variable reference
    if isinstance(exp, str):
        try:
            return [env.find(exp)[exp], sig]
        except:
            try:
                return [env[exp], sig]
            except:
                return[exp, sig]
    # constant
    if not isinstance(exp, list):
        if isinstance(exp, int) or isinstance(exp, float):
            exp = _totensor(exp)
        return [exp, sig]
    op, addr, *args = exp
    # address
    if addr[0] == 'push-address':
        addr_op, *addr_args = addr
        proc, _ = evaluate(exp=addr_op, sig=sig, env=env)
        alpha_key, addr_ = addr_args
        alpha = env[alpha_key]
        eval_addr = proc(alpha, [addr_])
        # update env
        env.update({alpha_key: eval_addr})

    # definition
    if op == 'fn':
        (parms, body) = args
        return Procedure(parms, body, env, sig)
    # procedure call
    else:
        proc, sig = evaluate(exp=op, sig=sig, env=env)
        vals = [evaluate(arg, env)[0] for arg in args]
        return [proc(*vals), sig]

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self):
            return self
        else:
            self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env, sig):
        self.parms, self.body, self.env, self.sig = parms, body, env, sig
    def __call__(self, *args):
        return evaluate(exp=self.body, sig=self.sig, env=Env(self.parms, args, self.env))

def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():

    for i in range(1,14):
    #for i in range(13,14):
        # # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/tests/deterministic/test_{i}.daphne'])
        # ast_path = f'./jsons/tests/deterministic/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of posterior of program {}:'.format(i))

        print('\nSample of posterior of program {}:'.format(i))
        ast_path = f'./jsons/tests/deterministic/{i}.json'
        with open(ast_path) as json_file:
            exp = json.load(json_file)
        # print(exp)

        exp = exp[2:][0]
        ret, sig = evaluate(exp, sig={}, env=global_env)

        truth = load_truth(f'{program_path}/src/programs/tests/deterministic/test_{i}.truth')
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        print('FOPPL Tests passed')

        global_env.update({'alpha' : []})

    # for i in range(1,13):
    #     # # Note: this path should be with respect to the daphne path!
    #     # ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/tests/hoppl-deterministic/test_{i}.daphne'])
    #     # ast_path = f'./jsons/tests/hoppl-deterministic/{i}.json'
    #     # with open(ast_path, 'w') as fout:
    #     #     json.dump(ast, fout, indent=2)
    #     # print('\n\n\nSample of posterior of program {}:'.format(i))
    #
    #     ast_path = f'./jsons/tests/deterministic/{i}.json'
    #     with open(ast_path) as json_file:
    #         graph = json.load(json_file)
    #     # print(graph)
    #
    #     # ret = evaluate(exp)
    #     # try:
    #     #     assert(is_tol(ret, truth))
    #     # except:
    #     #     raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
    #
    #     print('Test passed')

    print('All deterministic tests passed')



def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-2

    for i in range(1,7):
        # # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/tests/probabilistic/test_{i}.daphne'])
        # ast_path = f'./jsons/tests/probabilistic/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of posterior of program {}:'.format(i))

        ast_path = f'./jsons/tests/deterministic/{i}.json'
        with open(ast_path) as json_file:
            graph = json.load(json_file)
        # print(graph)

        # stream = get_stream(exp)
        #
        # p_val = run_prob_test(stream, truth, num_samples)
        #
        # print('p value', p_val)
        # assert(p_val > max_p_value)

    print('All probabilistic tests passed')



if __name__ == '__main__':
    program_path = '/home/tonyjo/Documents/prob-prog/CS539-HW-5'

    run_deterministic_tests()
    # run_probabilistic_tests()


    # for i in range(1,4):
    #     print(i)
    #     exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(evaluate(exp))
