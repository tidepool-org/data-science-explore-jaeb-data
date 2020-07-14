"""
flatten_tidepool_json.py

Custom JSON flattening procedure for Tidepool data

Code expanded from: github.com/amirziai/flatten


"""

def check_if_numbers_are_consecutive(list_):
    """
    Returns True if numbers in the list are consecutive
    :param list_: list of integers
    :return: Boolean
    """
    return all((True if second - first == 1 else False for first, second in zip(list_[:-1], list_[1:])))


def _construct_key(previous_key, separator, new_key, replace_separators=None):
    """
    Returns the new_key if no previous key exists, otherwise concatenates
    previous key, separator, and new_key
    :param previous_key:
    :param separator:
    :param new_key:
    :param str replace_separators: Replace separators within keys
    :return: a string if previous_key exists and simply passes through the
    new_key otherwise
    """
    if replace_separators is not None:
        new_key = str(new_key).replace(separator, replace_separators)
    if previous_key:
        return u"{}{}{}".format(previous_key, separator, new_key)
    else:
        return new_key


def flatten_json(nested_dict, separator="_", root_keys_to_ignore=set(), replace_separators=None):
    """
    Flattens a dictionary with nested structure to a dictionary with no
    hierarchy
    Consider ignoring keys that you are not interested in to prevent
    unnecessary processing
    This is specially true for very deep objects
    :param nested_dict: dictionary we want to flatten
    :param separator: string to separate dictionary keys by
    :param root_keys_to_ignore: set of root keys to ignore from flattening
    :param str replace_separators: Replace separators within keys
    :return: flattened dictionary
    """
    assert isinstance(nested_dict, dict), "flatten requires a dictionary input"
    assert isinstance(separator, str), "separator must be string"

    # This global dictionary stores the flattened keys and values and is
    # ultimately returned
    flattened_dict = dict()

    def _flatten(object_, key):
        """
        For dict, list and set objects_ calls itself on the elements and for
        other types assigns the object_ to
        the corresponding key in the global flattened_dict
        :param object_: object to flatten
        :param key: carries the concatenated key for the object_
        :return: None
        """
        is_ignore_root = False
        is_ignore_child = False

        # Check for ignore root / children
        if key:
            is_ignore_root = any([ignore_root == key for ignore_root in root_keys_to_ignore])
            is_ignore_child = any([ignore_root in key for ignore_root in root_keys_to_ignore])

        if not object_ or is_ignore_root:
            flattened_dict[key] = object_
        elif is_ignore_child:
            pass
        # These object types support iteration
        elif isinstance(object_, dict):
            for object_key in object_:
                if not (not key and object_key in root_keys_to_ignore):
                    _flatten(
                        object_[object_key],
                        _construct_key(key, separator, object_key, replace_separators=replace_separators),
                    )
                else:
                    flattened_dict[object_key] = object_
        elif isinstance(object_, (list, set, tuple)):
            for index, item in enumerate(object_):
                _flatten(item, _construct_key(key, separator, index, replace_separators=replace_separators))
        # Anything left take as is
        else:
            flattened_dict[key] = object_

    _flatten(nested_dict, None)
    return flattened_dict
