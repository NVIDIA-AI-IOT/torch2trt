import pytest
import torch
from torch2trt.flattener import Flattener


def test_flattener_from_value():

    x = (torch.ones(3), torch.ones(3))

    flattener = Flattener.from_value(x)

    assert(isinstance(flattener.schema, tuple))
    assert(flattener.schema[0] == 0)
    assert(flattener.schema[1] == 1)


def test_flattener_tuple():

    x = (torch.ones(3), torch.ones(3))

    flattener = Flattener.from_value(x)

    y = flattener.flatten(x)

    assert(len(y) == len(x))
    assert(y[0] is x[0])
    assert(y[1] is x[1])

    z = flattener.unflatten(y)

    assert(isinstance(z, tuple))
    assert(z[0] is x[0])
    assert(z[1] is x[1])


def test_flattener_list():
    
    x = [torch.ones(3), torch.ones(3)]

    flattener = Flattener.from_value(x)

    y = flattener.flatten(x)

    assert(len(y) == len(x))
    assert(y[0] is x[0])
    assert(y[1] is x[1])

    z = flattener.unflatten(y)

    assert(isinstance(z, list))
    assert(z[0] is x[0])
    assert(z[1] is x[1])


def test_flattener_dict():
    
    x = {'a': torch.ones(3), 'b': torch.ones(3)}

    flattener = Flattener.from_value(x)

    y = flattener.flatten(x)

    assert(len(y) == len(x))
    assert((y[0] is x['a'] and y[1] is x['b']) or (y[1] is x['a'] and y[0] is x['b']))

    z = flattener.unflatten(y)

    assert(isinstance(z, dict))
    assert(z['a'] is x['a'])
    assert(z['b'] is x['b'])


def test_flattener_nested_tuple():

    x = (torch.ones(1), (torch.ones(2), torch.ones(3)))

    flattener = Flattener.from_value(x)

    y = flattener.flatten(x)

    assert(len(y) == 3)
    
    z = flattener.unflatten(y)

    assert(isinstance(z, tuple))
    assert(isinstance(z[1], tuple))
    assert(z[0] is x[0])
    assert(z[1][0] is x[1][0])
    assert(z[1][1] is x[1][1])


def test_flattener_nested_list():

    x = [torch.ones(1), [torch.ones(2), torch.ones(3)]]

    flattener = Flattener.from_value(x)

    y = flattener.flatten(x)

    assert(len(y) == 3)
    
    z = flattener.unflatten(y)

    assert(isinstance(z, list))
    assert(isinstance(z[1], list))
    assert(z[0] is x[0])
    assert(z[1][0] is x[1][0])
    assert(z[1][1] is x[1][1])
    assert(z[0] is x[0])
    assert(z[1][0] is x[1][0])
    assert(z[1][1] is x[1][1])


def test_flattener_nested_dict():
    
    x = {'a': torch.ones(1), 'b': {'a': torch.ones(2), 'b': torch.ones(3)}}

    flattener = Flattener.from_value(x)

    y = flattener.flatten(x)

    assert(len(y) == 3)

    z = flattener.unflatten(y)

    assert(isinstance(z, dict))
    assert(isinstance(z['b'], dict))
    assert(z['a'] is x['a'])
    assert(z['b']['a'] is x['b']['a'])
    assert(z['b']['b'] is x['b']['b'])


def test_flattener_heterogeneous():

    x = {
        'a': (torch.ones(1), {'a': torch.ones(2)}),
        'b': [torch.ones(3), torch.ones(4), (torch.ones(5), {'a': torch.ones(6)})]
    }

    flattener = Flattener.from_value(x)

    y = flattener.flatten(x)

    assert(len(y) == 6)

    z = flattener.unflatten(y)

    assert(isinstance(z, dict))
    assert(isinstance(z['a'], tuple))
    assert(z['a'][0] is x['a'][0])
    assert(isinstance(z['a'][1], dict))
    assert(z['a'][1]['a'] is x['a'][1]['a'])
    assert(isinstance(z['b'], list))
    assert(z['b'][0] is x['b'][0])
    assert(z['b'][1] is x['b'][1])
    assert(isinstance(z['b'][2], tuple))
    assert(z['b'][2][0] is x['b'][2][0])
    assert(isinstance(z['b'][2][1], dict))
    assert(z['b'][2][1]['a'] is x['b'][2][1]['a'])