import pytest

import torch2trt.version_utils

def test_version_utils():
    
    a = torch2trt.version_utils.Version("10.1")

    assert a >= "10.1"
    assert a >= "10.0"
    assert a > "7.0"
    assert a < "11.0"
    assert a == "10.1"
    assert a <= "10.1"
    assert a <= "10.2"