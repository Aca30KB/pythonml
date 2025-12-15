def reduce(s):
    i = 0
    l = len(s)
    if l < 2:
        return s
    if l == 2 and s[0] == s[1]:
        return ""

    while i < l - 1:
        if s[i] == s[i + 1]:
            s = s[:i] + s[i + 2:]
            l = len(s)
            i = 0
        else:
            i += 1
    return s


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python reduce.py <string>")
        sys.exit(1)
    input_string = sys.argv[1]
    result = reduce(input_string)
    print(result)
    sys.exit(0)
# This code defines a function to reduce a string by removing adjacent duplicate characters.
# It also includes a command-line interface to test the function with user input.
