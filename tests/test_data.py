from pycrograd import datasets


def test_data_loading() -> None:
    data = datasets.MoonsData(100)
    assert len(data.data)
