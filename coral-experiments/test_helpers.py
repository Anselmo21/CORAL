from search.rank import _build_input_df, _build_input_df_from_voters


def test_build_input_df():
    voter_rankings = {
        "c1, c2": ["A", "B", "C"],
        "c1, c3": ["A", "B", "C"],
        "c2, c3": ["A", "B", "C"],
    }
    answers = ["A", "B", "C"]
    input_df = _build_input_df_from_voters(voter_rankings, answers, dataset_label="era")
    print(input_df)


if __name__ == "__main__":
    test_build_input_df()