def overrides(interface_class):
    """Simple decorator providing functionality like Java's @Override.

    This is useful for two main reasons:
        1. Ensure that signatures match after an interface method is refactored.
        2. Make it clear where an abstract method that is being implemented was
           originally inherited from.
    """
    def overrider(method):
        assert method.__name__ in dir(interface_class)
        return method
    return overrider
