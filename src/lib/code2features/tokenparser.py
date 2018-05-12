def tokenize(s):
    """
    :param s: Input text
    :return: list of strings
    :type s: basestring
    :rtype: list[str]
    """
    tokens = []
    offset = 0
    while offset < len(s):
        if s[offset].isalpha() or s[offset] == "_":
            right = offset
            while right < len(s) and (s[right].isalnum() or s[right] == "_"):
                right += 1
            tokens.append(s[offset:right])
            offset = right
        elif s[offset].isdigit():
            right = offset
            while right < len(s) and s[right].isdigit():
                right += 1
            tokens.append(s[offset:right])
            offset = right
        else:
            tokens.append(s[offset:offset + 1])
            offset += 1

    return tokens
