# 1. BETTER COMMENTS (Different colors and styles for different types of comments)

# Regular comment (standard gray)
# ! Alert / Warning: Critical logic, do not change without review (Red)
# ? Question: Why are we using a 30s timeout here? (Blue)
# TODO: Refactor this into a helper function (Orange)
# * Highlight / Note: Important method context (Green)
# // Deprecated / Strikethrough comment (Crossed out)

# 2. RAINBOW INDENTATION (Different colors for different levels of indentation)
if True:
    # * Level 1 indentation (Yell)
    if True:
        # * Level 2 indentation (Green)
        if True:
            # * Level 3 indentation (Pink)
            pass


# Auto Docstring Generation: Use docstring comments to describe functions, classes, and modules.
# *These comments are used by documentation generators and IDEs to provide context and usage information.
# * Just type triple quotes (""") and press Enter to auto-generate a docstring template.
def calculate_metrics(total_sales, tax_rate, discount=0.05):
    """_summary_

    Args:
        total_sales (_type_): _description_
        tax_rate (_type_): _description_
        discount (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    net_price = total_sales * (1 - discount)
    total_with_tax = net_price * (1 + tax_rate)

    summary_dict = {
        "original": total_sales,
        "net": net_price,
        "tax": total_with_tax,
        "status": "success",
    }

    return summary_dict


# 4. RUFF (Python linter) integration
# ! WINDOWS SHORTCUT: Press Ctrl + S to auto-format and clean on save
# * MANUAL FORMAT SHORTCUT: Press Shift + Alt + F to format without saving
# ? QUESTION: What happens to unused imports and messy spaces?
# TODO: Watch this whole file snap into clean, standard code!


def calculate_revenue(total_units, price_per_unit, discount=0.05):
    # ! Intentional messy formatting below:
    discount_price = price_per_unit * (1 - discount)

    total_revenue = discount_price * total_units
    summary_dict = {"original": total_units, "net": total_revenue, "status": "success"}

    return summary_dict


very_long_list_of_numbers = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
]

print(calculate_metrics(1250.50, 0.18))
