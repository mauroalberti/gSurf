
def is_number(s):
    """
    Check if string can be converted to number.

    @param  s:  parameter to check.
    @type  s:  string

    @return:  boolean, whether string can be converted to a number (float).

    """
    try:
        float(s)
    except:
        return False
    else:
        return True