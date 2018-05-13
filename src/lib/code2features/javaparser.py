from collections import deque
from token import TokenType


char_mapping = {
    ' ': TokenType.is_space,
    '\n': TokenType.is_newline,
    '\r': TokenType.is_newline,

    '{': TokenType.is_bracket_s_1,
    '[': TokenType.is_bracket_s_2,
    '(': TokenType.is_bracket_s_3,

    '}': TokenType.is_bracket_e_1,
    ']': TokenType.is_bracket_e_2,
    ')': TokenType.is_bracket_e_3,

    '.': TokenType.is_dot,
    ',': TokenType.is_comma,
    ';': TokenType.is_dotcomma,

    '@': TokenType.is_dog
}

op_mapping = {
    'if': TokenType.is_op_if,
    'for': TokenType.is_op_for,
    'while': TokenType.is_op_while,
    'try': TokenType.is_op_try,
    'catch': TokenType.is_op_catch,
    'finally': TokenType.is_op_finally,
    'final': TokenType.is_op_final,
    'assert': TokenType.is_op_assert,
    'null': TokenType.is_op_null,

    'public': TokenType.is_op_public,
    'private': TokenType.is_op_private,
    'protected': TokenType.is_op_protected,

    'void': TokenType.is_op_void,

    'byte': TokenType.is_type_byte,
    'short': TokenType.is_type_short,
    'int': TokenType.is_type_int,
    'long': TokenType.is_type_long,
    'float': TokenType.is_type_float,
    'double': TokenType.is_type_double,
    'char': TokenType.is_type_char,
    'String': TokenType.is_type_string,
    'boolean': TokenType.is_type_boolean,
}


def parse_number(tokens):
    if tokens[0].isdigit():
        tokens.popleft()
    elif len(tokens) >= 2 and tokens[0] == "-" and tokens[1].isdigit():
        tokens.popleft()
        tokens.popleft()
    else:
        return None
    token = TokenType.is_numeric
    if len(tokens) >= 2 and tokens[0] == "." and tokens[1].isdigit():
        tokens.popleft()
        tokens.popleft()
        token = TokenType.is_float_numeric
    return token


def parse_char(tokens):
    res = char_mapping.get(tokens[0])
    if res is not None:
        tokens.popleft()
    return res


def parse_op(tokens):
    res = op_mapping.get(tokens[0])
    if res is not None:
        tokens.popleft()
    return res


def parse_text(tokens):
    if tokens[0] == "\'" or tokens[0] == '\"':
        end = tokens[0]
        skip = True
        while tokens:
            if skip:
                tokens.popleft()
                skip = False
            elif tokens[0] == "\\":
                skip = True
                continue
            elif tokens[0] == end:
                tokens.popleft()
                break
            else:
                tokens.popleft()
        return TokenType.is_text


def parse_comment_oneline(tokens):
    if len(tokens) < 2:
        return None
    if tokens[0] == "/" and tokens[1] == "/":
        tokens.popleft()
        tokens.popleft()
        while tokens:
            token = tokens.popleft()
            if char_mapping.get(token) == TokenType.is_newline:
                break
        return TokenType.is_short_comment


def parse_comment_multiline(tokens):
    if len(tokens) <= 2:
        return None
    if tokens[0] == "/" and tokens[1] == "*":
        tokens.popleft()
        tokens.popleft()
        while tokens:
            if len(tokens) >= 2:
                if tokens[0] == "*" and tokens[1] == "/":
                    tokens.popleft()
                    tokens.popleft()
                    break
            tokens.popleft()
        return TokenType.is_long_comment


def word_classify(tokens):
    normalized_lower = tokens[0].replace("_", "x")
    if not normalized_lower.isalnum():
        return None
    token = tokens.popleft()
    if "_" not in token:
        if len(token) <= 2:
            return TokenType.is_very_short_word
        elif len(token) <= 7 and token.isupper() or token.islower():
            return TokenType.is_short_word
        elif token.isupper() or token.islower():
            return TokenType.is_long_lower_word if token.islower() else TokenType.is_long_upper_word
    elif token.isupper() or token.islower():
        if token[0] == "_":
            return TokenType.is_upper_case_word_ if token.isupper() else TokenType.is_lower_case_word_
        else:
            return TokenType.is_upper_case_word if token.isupper() else TokenType.is_lower_case_word
    return TokenType.is_camel_case_word


parsers = [
    parse_number,
    parse_char,
    parse_op,
    parse_text,
    parse_comment_oneline,
    parse_comment_multiline,
    word_classify
]


def parse(ts, with_shit=True):
    """
    :type ts: list[str]
    :param ts: list of raw tokens
    :param with_shit: allow shit-tokens
    :return: list of parse Tokens
    :rtype: list[Token]
    """
    tokens = deque(ts)

    ans = []
    while tokens:
        token_type = None
        for parser in parsers:
            token_type = parser(tokens)
            if token_type is not None:
                break
        if token_type is not None:
            ans.append(token_type)
        else:
            tokens.popleft()
            if with_shit:
                ans.append(TokenType.is_shit)

    if with_shit:
        return ans
    else:
        return [x for x in ans if x != TokenType.is_shit]
