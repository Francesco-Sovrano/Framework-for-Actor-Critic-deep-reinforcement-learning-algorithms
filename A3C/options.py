# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import configparser
import os
import ast

_parser = argparse.ArgumentParser()
_parser.add_argument('--configs', '-c', default="default_cfg.cfg", help="configuration file")
args = _parser.parse_args()

if not os.path.isfile(args.configs):
    raise ValueError("Could not find configuration file %s "
                     % os.path.join(os.path.realpath(os.path.curdir), args.configs))

cfg = configparser.ConfigParser()
cfg.read(args.configs)


def _parse_math_expression(expression):
    try:
        tree = ast.parse(expression, mode='eval')
        if not all(isinstance(node, (ast.Expression,
                                     ast.UnaryOp, ast.unaryop,
                                     ast.BinOp, ast.operator,
                                     ast.Num)) for node in ast.walk(tree)):
            raise ValueError()
        result = eval(compile(tree, filename='', mode='eval'))
        return result
    except SyntaxError:
        raise ValueError()


def _flatten_cfg(cfg):
    """
    :param configparser.ConfigParser cfg:
    :rtype: dict
    """
    d = {}
    for sec in cfg.sections():
        for key, val in cfg.items(sec):
            try:
                d[key] = ast.literal_eval(val)
            except ValueError:
                try:
                    d[key] = _parse_math_expression(val)
                except ValueError:
                    raise ValueError('Invalid value for key "%s": "%s"' % (key, val))
    return d


class Flags:
    @staticmethod
    def from_dict(dict):
        print("the dict is:::", dict)
        flags = Flags()
        for key, val in dict.items():
            setattr(flags, key, val)
        return flags


flags = Flags.from_dict(_flatten_cfg(cfg))
