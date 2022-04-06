class SafeDict(dict):
    # for generating latex tables
    def __missing__(self, key):
        return '{' + key + '}'
