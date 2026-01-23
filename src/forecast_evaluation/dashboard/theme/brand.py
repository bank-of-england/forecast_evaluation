from importlib.resources import files
from shiny import ui


def load_brand():
    """Load the Brand instance from _brand.yml"""
    brand_file = files(__package__) / "_brand.yml"

    return ui.Theme.from_brand(brand_file)


# Export for easy import
brand = load_brand()
