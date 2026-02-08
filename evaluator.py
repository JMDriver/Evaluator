"""Evaluator.

A left to right mathematical expression evaluator.

1) Open a root scope. A scope represents a block of the expression that is
either inside parenthesis, or, the root scope for the entire expression.

2) Iterate the characters in the expression.
    - If the character is a valid integer, append it to the scopes number
    buffer.
        - Any future non-integer characters flush this number buffer into an
        integer and apply pending operations.
    - If the character is a whitespace, skip the character.
    - If the character is a valid operator, set the scopes pending operator.
    - If the character is a '(', start a new scope.
    - If the character is a ')", merge the scope with its parent.
    - Else the character is invalid, return None.

3) Ensure just the root scope remains. Flush any remaining numbers
and apply pending operators. Return the root scopes value if valid.
"""

import dataclasses
import sys
import typing as t

# Define Operator as a Literal for strict type checking on keys.
Operator = t.Literal["+", "-", "/", "*"]

# Define a map for operator execution.
operator_map: t.Final[dict[Operator, t.Callable[[int, int], int | None]]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    # The outcome of evaluate should be an integer so, it is assumed all
    # operations should return integers. Additionally, we need to handle
    # division by 0.
    "/": lambda a, b: a // b if b else None,
    "*": lambda a, b: a * b,
}


# Given operator_map enforces Operator as keys, I can add a helper function
# that allows us to narrow characters to Operators with type-safety and MyPy
# will treat it correctly.
def is_operator(test_char: str, /) -> t.TypeGuard[Operator]:
    """Type-safe helper to check if a character is a valid Operator.

    Args:
        test_char: The character that should be checked as an operator.

    Returns:
        bool: True, if the test_char is an operator.

    """
    return test_char in operator_map


@dataclasses.dataclass(slots=True)
class Scope:
    """Track information for a scope.

    Args:
        accumulated_value: The accumulated value of this scope.
        operator: The current pending operator in this scope.

    Attributes:
        pending_number: The current number being parsed.

    """

    accumulated_value: int
    operator: Operator | None = None

    _number_buffer: list[str] | None = None

    @property
    def pending_number(self) -> int | None:
        """The current number being parsed.

        Returns:
            int | None: The pending number being parsed, or, None if no number
            is currently being parsed.

        """
        if self._number_buffer is None:
            return None

        return int("".join(self._number_buffer))

    def add_integer(self, test_char: str, /) -> bool:
        """Concatenate integer to the currently parsed number.

        Returns:
            bool: True, if the character is a valid integer.

        """
        if not ("0" <= test_char <= "9"):
            # Do direct comparison here, isnumeric() would be nicer but allows
            # invalid tokens.
            # If the character is not valid, early exit.
            return False

        self._number_buffer = self._number_buffer or []
        self._number_buffer.append(test_char)
        return True

    def clear_integer(self) -> None:
        """Clear all pending integers."""
        self._number_buffer = None


T = t.TypeVar("T")


# Use Generics to ensure typechecking.
# Very basic stack wrapper, used to make intention clearer by encapsulating
# list operations, and removing the [-1] magic number from evaluate.
class Stack(t.Generic[T]):
    """Basic stack implementation to encapsulate list operations."""

    __slots__ = ("_array",)

    def __init__(self) -> None:
        """Initialize the backing array."""
        self._array: t.Final[list[T]] = []

    def push(self, item: T, /) -> None:
        """Push to the stack."""
        self._array.append(item)

    def pop(self) -> T:
        """Pop from the stack."""
        return self._array.pop()

    def back(self) -> T:
        """Peek at the last element in the stack."""
        return self._array[-1]

    def empty(self) -> bool:
        """Return True if the stack is empty."""
        return not len(self)

    def __len__(self) -> int:
        """Return the size of the state."""
        return len(self._array)


# Use int | None for the return signature instead of t.Optional[int], this does
# change the function signature slightly from the Task, however,
# python 3.10+ expects this as the standard and the behaviour is identical.
def evaluate(expression: str, /) -> int | None:
    """Evaluate a mathematical expression.

    Returns:
        int | None: The result if the expression is valid.

    """
    scope_stacks: t.Final[Stack[Scope]] = Stack()

    # The root scope will always start with a number or bracket.
    # This means a scope is being entered, and the first number should be added
    # to 0.
    scope_stacks.push(Scope(0, operator="+"))

    # Iterate chars and parse them, if they are invalid return None.
    for test_char in expression:
        if not _process_char(test_char, scope_stacks):
            return None

    if len(scope_stacks) != 1:
        # Scopes should have collapsed. Mismatched parenthesis. e.g: ((1+2).
        return None

    final_scope: t.Final = scope_stacks.back()

    if not _flush_buffered_number(final_scope):
        return None

    if final_scope.operator:
        # Expression ends in an operator. e.g: 1+2+
        return None

    return final_scope.accumulated_value


def _close_scope(scope_stacks: Stack[Scope]) -> bool:
    """Finalize the current scope and merge it into its parent.

    Returns:
        bool: True if successful.

    """
    current_scope: t.Final = scope_stacks.pop()

    if len(scope_stacks) == 0:
        # No parent to merge with, too many closing parentheses. e.g: (1+1))
        return False

    if current_scope.operator is not None:
        # Scope ends in an operator, this is invalid. e.g: 1+
        return False

    parent_scope = scope_stacks.back()
    if not parent_scope.operator:
        # If the parent has no operator then we have an invalid
        # expression. e.g: 1(2+2)
        # This should be implicit multiplication but it does not
        # seem to fit the task.
        return False

    if (
        result := operator_map[parent_scope.operator](
            parent_scope.accumulated_value,
            current_scope.accumulated_value,
        )
    ) is None:
        # Invalid operation. e.g., division by 0.
        return False

    parent_scope.accumulated_value = result
    parent_scope.operator = None
    return True


def _flush_buffered_number(scope: Scope, /) -> bool:
    """Apply the pending operator to a parsed number.

    Returns:
        bool: True if the expression is valid.

    """
    number: t.Final = scope.pending_number

    if number is None:
        return True

    if scope.operator:
        if (
            result := operator_map[scope.operator](
                scope.accumulated_value,
                number,
            )
        ) is None:
            # The operation returned an invalid result. e.g 1/0
            return False

        scope.accumulated_value = result
        scope.operator = None
        scope.clear_integer()
        return True
    # If we have no pending operator, the expression is invalid. e.g:
    # (5 5)
    return False


def _process_char(
    test_char: str,
    scopes: Stack[Scope],
) -> bool:
    current_scope: t.Final = scopes.back()

    if current_scope.add_integer(test_char):
        # If this is a valid integer, add it to the scopes pending number list.
        return True

    if not _flush_buffered_number(current_scope):
        return False

    if test_char.isspace():
        # If whitespace we can skip, this is delayed because the whitespace
        # may end a numeric token.
        return True

    # Determine if the test_char is an operator.
    if is_operator(test_char):
        if current_scope.operator is not None:
            # Two consecutive operators (1++1) is invalid.
            return False
        # Set the scopes pending operator.
        current_scope.operator = test_char
        return True

    if test_char == "(":
        # A new scope is being entered, and the first number should be
        # added to 0.
        scopes.push(Scope(0, operator="+"))
        return True

    if test_char == ")":
        return _close_scope(scopes)

    return False


# -----------------------Test Runner ----------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TestCase:
    """Defines a single test scenario.

    Args:
        description: A brief summary of what this case tests.
        expected: The expected result, or None if an error is expected.
        expression: An expression string to evaluate.

    """

    description: str
    expected: int | None
    expression: str


def run_tests(test_cases: t.Iterable[TestCase]) -> None:
    """Run all tests cases."""
    failure_count: int = 0
    for test in test_cases:
        result: int | None = evaluate(test.expression)
        if result != test.expected:
            print(
                f"FAILED ({test.expression}): {test.description}. ({result} !="
                f" {test.expected}).",
            )
            failure_count += 1
            continue

        print(f"PASSED ({test.expression}): {test.description}.")

    if failure_count:
        sys.exit(1)


if __name__ == "__main__":
    tests: t.Final[tuple[TestCase, ...]] = (
        # ------------- Basic Operations --------------------------------
        # Using kwargs just at top of the test block so it is clear to see
        # the TestCase structure easily when viewing the test block.
        TestCase(
            description="Simple addition",
            expected=10,
            expression="5 + 5",
        ),
        TestCase(
            description="Simple division",
            expected=1,
            expression="5 / 5",
        ),
        TestCase(
            description="Simple multiplication",
            expected=25,
            expression="5 * 5",
        ),
        TestCase(
            description="Simple subtraction",
            expected=0,
            expression="5 - 5",
        ),
        TestCase(
            description="Floating-point truncation",
            expected=2,
            expression="5 / 2",
        ),
        TestCase(description="No operators", expected=1, expression="1"),
        # ----------- Left to right -------------------------------------
        TestCase("Left to right division and multiplication", 4, "8/2*1"),
        TestCase("Left to right addition and multiplication", 9, "1+2*3"),
        # ----------- Multi-digit and whitespace ------------------------
        TestCase("Double-digit numbers", 20, "10 + 10"),
        TestCase("Triple-digit numbers", 200, "100 * 2"),
        TestCase("Single whitespace handling spaces", 3, " 1 + 2 "),
        TestCase("ASCII whitespace handling", 3, " 1\n+\t2 "),
        TestCase("Mixed whitespace handling", 6, "1+2  * 2"),
        TestCase("Whitespace in parenthesis", 3, "(1 + 2)"),
        # ------------- Nesting -----------------------------------------
        TestCase("Simple nested operation", 4, "(1+3)"),
        TestCase("Operator before parenthesis", 8, "2 * (1+3)"),
        TestCase("Operator after parentheses", 4, "(1+1)*2"),
        TestCase("Two distinct scopes", 8, "(1+1) * (1+3)"),
        TestCase("Deeply nested scopes", 8, "(1+1) * (1+(1+2))"),
        TestCase("Root redundant matching parenthesis", 1, "(1)"),
        TestCase("Many redundant matching parenthesis", 1, "(((((1)))))"),
        TestCase(
            "Many redundant matching parenthesis with operator",
            2,
            "(((((1+1)))))",
        ),
        TestCase(
            "Many redundant matching parenthesis with external operator",
            4,
            "2*(((((1+1)))))",
        ),
        # ----------- Operators with 0 -------------------------------------
        # A statement such as:
        #
        #     if number:
        #         return
        #
        # Could confuse 0 and None if we are not explicit. Ensure this isn't
        # occuring.
        TestCase("Addition of 0", 1, "1 + 0"),
        TestCase("Addition to 0", 1, "0 + 1"),
        TestCase("0 only addition", 0, "0+0"),
        TestCase("Leading 0 addition", 1, "0001+0000"),
        TestCase("0 in parenthesis", 0, "(0)"),
        TestCase("Leading 0 in parenthesis", 0, "(0000)"),
        # ----------- Division by 0 -------------------------------------
        TestCase("Division by 0", None, "1/0"),
        TestCase("Zero divided by 0", None, "0/0"),
        TestCase("Division by evaluated zero", None, "1/(2-2)"),
        TestCase(
            "Division by 0, where 0 has leading characters",
            None,
            "1/00000",
        ),
        TestCase(
            "Deeply nested scopes with division by 0",
            None,
            "((1+1) * (1+(1+2))) / 0",
        ),
        # ----------- Invalid characters --------------------------------
        TestCase("Invalid operator", None, "1=2"),
        TestCase("Unsigned numbers", None, "-1"),
        TestCase("Unsigned number pre-operator", None, "-1*2"),
        TestCase("Unsigned number post-operator", None, "1*-2"),
        TestCase("Unsigned parenthesis", None, "-(1*2)"),
        TestCase("Unsigned number in parenthesis", None, "(-1)"),
        TestCase(
            "Unsigned number in parenthesis pre-operator", None, "(-1*2)",
        ),
        TestCase(
            "Unsigned number in parenthesis post-operator", None, "(1*-2)",
        ),
        # isnumeric() would allow this, we should not.
        TestCase("Global numeric characters", None, "五-五"),
        TestCase("Invalid character in expression", None, "2,"),
        TestCase("Decimal Numbers", None, "2.5"),
        TestCase("Decimal Numbers with operator", None, "2.5 + 3"),
        TestCase("Superscript character", None, "²"),
        TestCase("Superscript character with valid numeric", None, "2²"),
        TestCase("Superscript character with valid operator", None, "2*²"),
        TestCase("Fractional characters", None, "½"),
        TestCase("Fractional characters with valid numeric", None, "2½"),
        TestCase("Fractional characters with valid operator", None, "2*½"),
        # ----------- Empty expressions --------------------------------
        # These are deemed invalid, however, handling can be added, it needs to
        # be considered if these are invalid or 0.
        TestCase("Empty expression", None, ""),
        TestCase("Empty expression with whitespace", None, "   "),
        TestCase("Nested empty expression", None, "()"),
        TestCase("Nested empty expression with whitespace", None, " ( ) "),
        # ----------- Invalid Operator --------------------------------
        TestCase("Expression starts with an operator", None, "+1"),
        TestCase("Single operator only", None, "+"),
        TestCase("Missing Operator", None, "1 1"),
        TestCase("Nested Missing Operator", None, "(1 1)"),
        TestCase("Expression ends with an operator", None, "1+"),
        TestCase("Parenthesis starts with an operator", None, "(+1)"),
        TestCase("Parenthesis ends with an operator", None, "(1+)"),
        TestCase("Multiple concurrent operators", None, "1++1"),
        TestCase(
            "Multiple concurrent operators with whitespace",
            None,
            "1 + + 1",
        ),
        # ----------- Invalid Parenthesis --------------------------------
        # Technically correct - implicit multiplication, but not supported.
        TestCase("Implicit multiplication", None, "1(2*3)"),
        TestCase("Parenthesis Implicit multiplication", None, "(2*3)(2*3)"),
        TestCase("Invalid Implicit multiplication", None, "(2*3)4"),
        TestCase("Unclosed root parenthesis start", None, "(4"),
        TestCase("Unclosed root parenthesis end", None, "4)"),
        TestCase("Unclosed nested parenthesis start", None, "((4)"),
        TestCase("Unclosed nested parenthesis end", None, "(4))"),
        TestCase("Inverse parenthesis", None, ")4("),
    )

    run_tests(tests)
