import pandas as pd  # noqa: TID253
import pytest
from sqlglot import exp

from sqlmesh.core.model.seed import CsvSettings, Seed


def test_read():
    content = """key,value,ds,bool
1,one,2022-01-01,true
2,two,2022-01-02,false
3,three,2022-01-03,true
"""
    seed = Seed(content=content)
    # Since we provide "snowflake" as the dialect, all identifiers are expected to
    # be normalized according to its resolution rules, hence the uppercase names
    seed_reader = seed.reader(dialect="snowflake")

    assert seed_reader.columns_to_types == {
        "KEY": exp.DataType.build("bigint"),
        "VALUE": exp.DataType.build("text"),
        "DS": exp.DataType.build("text"),
        "BOOL": exp.DataType.build("boolean"),
    }
    expected_df = pd.DataFrame(
        data={
            "KEY": [1, 2, 3],
            "VALUE": ["one", "two", "three"],
            "DS": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "BOOL": [True, False, True],
        }
    )
    dfs = seed_reader.read(batch_size=2)
    pd.testing.assert_frame_equal(next(dfs), expected_df.iloc[:2, :])
    pd.testing.assert_frame_equal(next(dfs), expected_df.iloc[2:, :])

    with pytest.raises(StopIteration):
        next(dfs)


def test_read_custom_settings():
    content = """key,value,ds
1,'one','2022-01-01'
2,'two','2022-01-02'
3,'three','2022-01-03'
"""
    seed = Seed(content=content)
    seed_reader = seed.reader(settings=CsvSettings(quotechar="'"))

    expected_df = pd.DataFrame(
        data={
            "key": [1, 2, 3],
            "value": ["one", "two", "three"],
            "ds": ["2022-01-01", "2022-01-02", "2022-01-03"],
        }
    )
    dfs = seed_reader.read()
    pd.testing.assert_frame_equal(next(dfs), expected_df)


def test_read_returns_independent_batches():
    content = """key,value
1,one
2,two
"""
    seed = Seed(content=content)
    seed_reader = seed.reader()

    # Keep the generator open so the copy_on_write context inside read() stays active.
    gen = seed_reader.read(batch_size=1)
    first_batch = next(gen)
    # Mutate while the generator (and therefore the CoW context) is still open.
    # CoW ensures only first_batch gets a private copy; the cached _df is unchanged.
    first_batch.at[0, "value"] = "changed"
    # second_batch is fetched while CoW is still active, so it still sees the original data.
    second_batch = next(gen)

    assert first_batch["value"].tolist() == ["changed"]
    assert second_batch["value"].tolist() == ["two"]
    # CoW prevented the mutation from reaching the cached _df, so a fresh read returns original data.
    assert next(seed_reader.read())["value"].tolist() == ["one", "two"]


def test_column_hashes():
    content = """key,value,ds
1,one,2022-01-01
2,two,2022-01-02
3,three,2022-01-03
"""
    seed = Seed(content=content)
    assert seed.reader().column_hashes == {
        "key": "122302783",
        "value": "1969959181",
        "ds": "725407375",
    }

    content_column_changed = """key,value,ds
1,one,2022-01-01
2,two,2022-01-05
3,three,2022-01-03

"""
    seed_column_changed = Seed(content=content_column_changed)
    assert seed_column_changed.reader().column_hashes == {
        **seed.reader().column_hashes,
        "ds": "3396890652",
    }
