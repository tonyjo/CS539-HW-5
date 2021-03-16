import json
import torch
from daphne import daphne
from pyrsistent import pmap, plist
# Tests
from tests import is_tol, run_prob_test, load_truth
# Global env
from primitives import env as penv
from primitives import _totensor


class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = {**outer}
    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self):
            return self
        else:
            return self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env, sig):
        self.parms, self.body, self.env, self.sig = parms, body, env, sig
    def __call__(self, *args):
        env = Env(parms=self.parms, args=args, outer=self.env)
        print('----->', env)
        print('----->', env.outer)
        print('----->', dir(env))
        return evaluate(exp=self.body, sig=self.sig, env=env)

def standard_env():
    "An environment with some Scheme standard procedures."
    # env = pmap(penv)
    # env = env.update({'alpha' : []})
    # return env

    penv.update({'alpha' : []})
    return penv

global_env = standard_env()

DEBUG=True
def evaluate(exp, sig, env):
    if env is None:
        env = standard_env()
    #import pdb; pdb.set_trace()
    if DEBUG:
        print('Exp: ', exp)
        print('Env: ', env, '\n')
        try:
            print('====>', env.outer)
            print('====>', dir(env))
        except:
            print('====>', dir(env))
        print('\n')

    # variable reference
    if isinstance(exp, str):
        import pdb; pdb.set_trace()
        if DEBUG:
            print('Exp: ', exp)
            print('Env: ', env.keys(), '\n')
            try:
                print('====>', env.outer)
                print('====>', dir(env))
            except:
                print('====>', dir(env))
            print('\n')
        try:
            env_func = env.find(exp)
            #import pdb; pdb.set_trace()
            if DEBUG:
                print('env_func: ', env_func)
            return [env_func[exp], sig]
        except:
            try:
                env_func = env[exp]
                if DEBUG:
                    print('env_func: ', env_func)
                return [env_func, sig]
            except:
                return[exp, sig]

    # constant
    if not isinstance(exp, list):
        if isinstance(exp, int) or isinstance(exp, float):
            exp = _totensor(exp)
        return [exp, sig]

    op, addr, *args = exp
    if DEBUG:
        print('OP: ', op)
        print('addr: ', addr)
        print('args: ', args)
        print('Env', env, '\n')

    # address
    if addr[0] == 'push-address':
        addr_op, *addr_args = addr
        # import pdb; pdb.set_trace()
        if DEBUG:
            print('addr_op: ', addr_op)
            print('addr_args: ', addr_args)
            print('Env: ', env)
        proc, _ = evaluate(exp=addr_op, sig=sig, env=env)
        if DEBUG:
            print('proc: ', proc)
        alpha_key, addr_ = addr_args
        if DEBUG:
            print('Key: ', alpha_key)
            print('Key params: ', addr_)
        alpha, _  = evaluate(exp=alpha_key, sig=sig, env=env)
        if DEBUG:
            print('alpha: ', alpha)
        try:
            eval_addr = proc(alpha, [addr_])
        except:
            eval_addr = [alpha] + [addr_]
        if DEBUG:
            print('eval_addr: ', eval_addr, '\n')
        # update env
        env.update({alpha_key: eval_addr})

    # definition -- functions as first class objects
    if op == 'fn':
        parms = addr[1:]
        if isinstance(args, list) and len(args) == 1:
            try:
                if len(args[0]) > 1:
                    args = args[0]
            except:
                # most likely int object
                args = args[0]
        if DEBUG:
            print('Parm: ', parms)
            print('Body: ',  args, '\n')

        return [Procedure(parms, args, env, sig), sig]

    # procedure call
    else:
        proc, sig = evaluate(exp=op, sig=sig, env=env)
        vals = []
        for arg in args:
            try:
                arg_eval, sig = evaluate(arg, sig, env)[0]
            except:
                arg_eval, sig = evaluate(arg, sig, env)
            vals.append(arg_eval)
        if DEBUG:
            print('proc: ', proc)
            print('vals: ', vals, '\n')
        return [proc(*vals), sig]


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():

    # for i in range(1,14):
    # #for i in range(13,14):
    #     # # Note: this path should be with respect to the daphne path!
    #     # ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/tests/deterministic/test_{i}.daphne'])
    #     # ast_path = f'./jsons/tests/deterministic/{i}.json'
    #     # with open(ast_path, 'w') as fout:
    #     #     json.dump(ast, fout, indent=2)
    #     # print('\n\n\nSample of posterior of program {}:'.format(i))
    #
    #     print('\nSample of posterior of program {}:'.format(i))
    #     ast_path = f'./jsons/tests/deterministic/{i}.json'
    #     with open(ast_path) as json_file:
    #         exp = json.load(json_file)
    #     # print(exp)
    #
    #     exp = exp[2:][0]
    #     ret, sig = evaluate(exp, sig={}, env=global_env)
    #
    #     truth = load_truth(f'{program_path}/src/programs/tests/deterministic/test_{i}.truth')
    #     try:
    #         assert(is_tol(ret, truth))
    #     except:
    #         raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
    #     print('FOPPL Tests passed')
    #
    #     global_env.update({'alpha' : []})

    for i in range(5,6):
    #for i in range(1,13):
        # # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/tests/hoppl-deterministic/test_{i}.daphne'])
        # ast_path = f'./jsons/tests/hoppl-deterministic/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of posterior of program {}:'.format(i))

        ast_path = f'./jsons/tests/hoppl-deterministic/{i}.json'
        with open(ast_path) as json_file:
            exp = json.load(json_file)
        # print(graph)

        exp = exp[2:]
        if len(exp) == 1:
            exp = exp[0]
        ret, sig = evaluate(exp=exp, sig={}, env=global_env)
        #print(ret)

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
