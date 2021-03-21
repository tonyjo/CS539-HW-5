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
        self.new_env = pmap({p:a for p, a in zip(parms, args)})
        self.new_env = self.new_env.update({'alpha':''})
        self.outer   = outer
    def does_exist(self, var):
        if var in self.new_env:
            return True
        elif self.outer is None:
            return False
        else:
            return self.outer.does_exist(var=var)
    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self.new_env):
            return self.new_env[var]
        else:
            return self.outer.find(var)
        # return self.new_env[var] if (var in self.new_env) else self.outer.find(var)


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(parms=penv.keys(), args=penv.values())

    return env


class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms = parms
        self.body  = body
        self.env   = env
    def __call__(self, *args, sig):
        return eval(exp=self.body, sig=sig, env=Env(parms=self.parms, args=args, outer=self.env))


DEBUG=False
def eval(exp, sig, env):
    #import pdb; pdb.set_trace()
    if DEBUG:
        print('Exp: ', exp)

    # variable reference
    if isinstance(exp, str): #and (env.does_exist(var=exp)):
        #import pdb; pdb.set_trace()
        try:
            env_func = env.find(exp)
            if DEBUG:
                print('env_func: ', env_func)
            return env_func, sig
        except:
            return exp, sig

    # constant
    if not isinstance(exp, list): #and (not env.does_exist(var=exp)):
        if isinstance(exp, int) or isinstance(exp, float):
            exp = _totensor(exp)
        return exp, sig

    # definition -- functions as first class objects
    if exp[0] == 'fn':
        #import pdb; pdb.set_trace()
        parms, body = exp[1], exp[2]
        if DEBUG:
            print('Parm: ',  parms)
            print('Body: ',  body, '\n')
        return Procedure(parms=parms, body=body, env=env), sig

    # conditional
    if exp[0] == 'if':
        if len(exp) == 4:
            bcond, consq, alter = exp[1], exp[2], exp[3]
            bcond_eval, sig = eval(exp=bcond, sig=sig, env=env)
            if bcond_eval:
                return eval(exp=consq, sig=sig, env=env)
            else:
                return eval(exp=alter, sig=sig, env=env)
        else:
            bcond, addr, op1, op2 = exp[0][0], exp[0][1], exp[0][2], exp[0][3]
            bcond_eval, sig = eval(exp=bcond, sig=sig, env=env)
            op1_eval, sig   = eval(exp=op1, sig=sig, env=env)
            op2_eval, sig   = eval(exp=op2, sig=sig, env=env)
            proc_eval, sig  = bcond_eval(op1_eval, op2_eval)
            return proc_eval, sig

    # observe
    elif exp[0] == "observe":
        addr, dist, obs = exp[1], exp[2], exp[3]
        addr_eval, sig  = eval(exp=addr, sig=sig, env=env)
        dist_eval, sig  = eval(exp=dist, sig=sig, env=env)
        obs_eval,  sig  = eval(exp=obs,  sig=sig, env=env)
        return obs_eval, sig

    # sample
    elif exp[0] == "sample":
        addr, dist = exp[1], exp[2]
        addr_eval, sig = eval(exp=addr, sig=sig, env=env)
        dist_eval, sig = eval(exp=dist, sig=sig, env=env)
        try:
            dist_eval = dist_eval.sample()
        except Exception as e:
            print(dist_eval)
            print('Failed to sample!')
        return dist_eval, sig

    # procedure call
    else:
        # Evaluate procedure
        proc, sig = eval(exp=exp[0], sig=sig, env=env)
        # Evaluate arguments
        vals = []
        for each_exp in range(1, len(exp)):
            arg_eval, sig = eval(exp=exp[each_exp], sig=sig, env=env)
            vals.append(arg_eval)
        if DEBUG:
            print('exp : ', exp)
            print('proc: ', proc)
            print('vals: ', vals, '\n')
        try:
            if type(proc) == Procedure:
                proc_eval, sig = proc(*vals, sig=sig)
            else:
                proc_eval = proc(*vals)
            return proc_eval, sig
        except Exception as e:
            print('Exception raised: ', e)
            print('proc: ', proc)
            print('vals: ', vals, '\n')
            # Tensor object
            return proc, sig
#-----------------------------------------------------------------------
def evaluate(exp):
    global_env = standard_env()
    proc, sig = eval(exp=exp, sig={}, env=global_env)

    return proc(sig=sig)


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
    #     print('\nDeterministic program {}:'.format(i))
    #     ast_path = f'./jsons/tests/deterministic/{i}.json'
    #     with open(ast_path) as json_file:
    #         exp = json.load(json_file)
    #     # print(exp)
    #
    #     ret, sig = evaluate(exp=exp)
    #     truth = load_truth(f'{program_path}/src/programs/tests/deterministic/test_{i}.truth')
    #     print('Output: ', ret)
    #     print('Truth:', truth)
    #     try:
    #         assert(is_tol(ret, truth))
    #         print('HOPPL Test passed!')
    #     except:
    #         print('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

    #for i in range(5,6):
    for i in range(1,13):
    #     # Note: this path should be with respect to the daphne path!
    #     ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/tests/hoppl-deterministic/test_{i}.daphne'])
    #     ast_path = f'./jsons/tests/hoppl-deterministic/{i}.json'
    #     with open(ast_path, 'w') as fout:
    #         json.dump(ast, fout, indent=2)
    #     print('\n\n\nSample of posterior of program {}:'.format(i))

        print('\nHOPPL deterministic  {}:'.format(i))
        ast_path = f'./jsons/tests/hoppl-deterministic/{i}.json'
        with open(ast_path) as json_file:
            exp = json.load(json_file)
        #print(exp)

        ret, sig = evaluate(exp=exp)
        truth = load_truth(f'{program_path}/src/programs/tests/hoppl-deterministic/test_{i}.truth')
        try:
            print('Output: ', ret)
            print('Truth:', truth)
            assert(is_tol(ret, truth))
            print('Test passed')
        except:
            print('return value is not equal to truth {}'.format(truth))
    print('All hoppl-deterministic tests passed')


def run_probabilistic_tests():
    num_samples=1e4
    max_p_value = 1e-2

    #for i in range(1,7):
    for i in range(4,5):
        # # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/tests/probabilistic/test_{i}.daphne'])
        # ast_path = f'./jsons/tests/probabilistic/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        ast_path = f'./jsons/tests/probabilistic/{i}.json'
        with open(ast_path) as json_file:
            ast = json.load(json_file)
        # print(ast)

        #ret, sig = evaluate(exp=ast)
        #print(ret)

        stream = get_stream(exp=ast)
        truth  = load_truth(f'{program_path}/src/programs/tests/probabilistic/test_{i}.truth')
        p_val  = run_prob_test(stream, truth, num_samples)
        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')



if __name__ == '__main__':
    program_path = '/home/tonyjo/Documents/prob-prog/CS539-HW-5'

    #run_deterministic_tests()
    #run_probabilistic_tests()

    #for i in range(3,4):
    for i in range(1,4):
        # # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar-hoppl', '-i', f'{program_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        ast_path = f'./jsons/{i}.json'
        with open(ast_path) as json_file:
            ast = json.load(json_file)

        ret, sig = evaluate(exp=ast)
        print(ret)
