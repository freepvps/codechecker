from token import TokenType
from tokenparser import tokenize
from javaparser import parse as parse_java


def extract_features(s):
    """
    :type s: basestring
    :param s: input code
    :return: features list
    :rtype: list[int]
    """
    return parse_java(tokenize(s))
