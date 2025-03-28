# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from app.utils.error_message import get_field_path


def test_empty_location():
    """Test with empty tuple"""
    assert get_field_path(()) == ""


def test_single_string():
    """Test with single string element"""
    assert get_field_path(("field",)) == "field"


def test_single_integer():
    """Test with single integer element"""
    assert get_field_path((0,)) == "[0]"


def test_multiple_strings():
    """Test with multiple string elements"""
    assert (
        get_field_path(("parent", "child", "grandchild")) == "parent.child.grandchild"
    )


def test_multiple_integers():
    """Test with multiple integer elements"""
    assert get_field_path((0, 1, 2)) == "[0][1][2]"


def test_mixed_types():
    """Test with mixture of strings and integers"""
    assert get_field_path(("array", 0, "field", 1)) == "array[0].field[1]"


def test_starting_with_integer():
    """Test path starting with integer"""
    assert get_field_path((0, "field", "subfield")) == "[0].field.subfield"


def test_complex_path():
    """Test complex path with multiple types"""
    assert (
        get_field_path(("users", 0, "addresses", 1, "street"))
        == "users[0].addresses[1].street"
    )
