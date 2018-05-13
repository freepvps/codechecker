field = "TOKEN_TYPE_FIELD"
ignorefield = "TOKEN_TYPE_FIELD_IGNORE"


class TokenType(object):
    is_numeric = field
    is_float_numeric = field

    is_space = field
    is_newline = field
    is_bracket_s_1 = field # {
    is_bracket_e_1 = field # }
    is_bracket_s_2 = field # [
    is_bracket_e_2 = field # ]
    is_bracket_s_3 = field # (
    is_bracket_e_3 = field # )
    is_dot = field # .
    is_comma = field # ,
    is_dotcomma = field # ;
    is_dog = field # @

    is_op_for = field
    is_op_if = field
    is_op_while = field
    is_op_try = field
    is_op_catch = field
    is_op_finally = field
    is_op_final = field
    is_op_assert = field
    is_op_null = field
    is_op_public = field
    is_op_private = field
    is_op_protected = field

    #is_type_byte = field
    #is_type_short = field
    #is_type_int = field
    #is_type_long = field
    #is_type_float = field
    #is_type_double = field
    #is_type_boolean = field
    #is_type_char = field
    #is_type_string = field
    is_type = field

    is_camel_case_word = field
    is_lower_case_word = field
    is_upper_case_word = field
    is_lower_case_word_ = field
    is_upper_case_word_ = field
    is_very_short_word = field # <= 2
    is_short_word = field # <= 7
    is_long_lower_word = field # > 7
    is_long_upper_word = field # > 7

    is_text = ignorefield

    is_short_comment = field
    is_long_comment = field

    is_shit = field
    size = 0
    names = {}
    for k, v in sorted(locals().iteritems()):
        if v == field:
            locals()[k] = size
            names[size] = k
            size += 1
    for k, v in sorted(locals().iteritems()):
        if v == ignorefield:
            locals()[k] = is_shit