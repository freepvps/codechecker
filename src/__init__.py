from tokenparser import tokenize
from javaparser import parse as parse_java
from token import TokenType

while True:
    s = raw_input()
    tokens = tokenize(s)
    print(tokens)
    parsed = parse_java(tokens)
    print([TokenType.names[x] for x in parsed])