# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from egqa import extract_state_change, create_belief_state_dictionary


def test_create_belief_state_dictionary():
    example = [['hotel-price range', 'cheap'], ['hotel-type', 'hotel']]
    result = {"hotel-price range": "cheap", "hotel-type": "hotel"}

    output = create_belief_state_dictionary(example)

    assert output == result


def test_extract_state_change():
    examples = [
        ([], [['hotel-price range', 'cheap'], ['hotel-type', 'hotel']]),
        (
            [['hotel-price range', 'cheap'], ['hotel-type', 'hotel']],
            [['hotel-price range', 'expensive'], ['hotel-name', 'sample']],
        ),
    ]
    results = [
        {"hotel-price range": ["INSERT", "cheap"], "hotel-type": ["INSERT", "hotel"]},
        {
            "hotel-price range": ["UPDATE", "expensive"],
            "hotel-type": ["DELETE", "hotel"],
            "hotel-name": ["INSERT", "sample"],
        },
    ]

    for ex, res in zip(examples, results):
        output = extract_state_change(
            previous_belief_state=ex[0], current_belief_state=ex[1]
        )
        assert output == res
