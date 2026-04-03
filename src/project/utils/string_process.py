def snake_to_camel(snake_str: str):
    components = snake_str.split("_")
    return components[0] + "".join(x.capitalize() for x in components[1:])


def snake_to_pascal(snake_str: str):
    return "".join(word.capitalize() for word in snake_str.split("_"))
